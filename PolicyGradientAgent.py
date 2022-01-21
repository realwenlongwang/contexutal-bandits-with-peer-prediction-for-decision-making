import numpy as np
from scipy.special import logit, expit
from scipy.ndimage import uniform_filter1d
from Environment import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


class Agent:

    def __init__(self, feature_num, action_num, learning_rate_theta, name, algorithm='regular'):
        self.theta_mean = np.zeros((feature_num * action_num, action_num))
        # self.theta_mean = np.random.uniform(low=-1.0, high=1.0, size=(feature_num * action_num, action_num))
        # self.theta_mean = np.random.normal(0, np.sqrt(2/(feature_num * action_num)), (feature_num * action_num, action_num))
        self.init_learning_rate_theta = learning_rate_theta
        self.learning_rate_theta = self.init_learning_rate_theta
        self.feature_num = feature_num
        self.action_num = action_num

        # Performance record
        self.evaluating = False
        self.evaluation_step = 1
        self.bucket_name_list = list('bucket' + str(i) for i in range(self.action_num))
        self.report_history_list = list()
        self.mean_gradients_history_list = list([] for i in range(action_num))
        self.mean_weights_history_list = list([] for i in range(action_num))
        self.reward_history_list = list()
        self.algorithm = algorithm
        self.name = name

    def save_weights(self, file_name):
        np.save(file_name, self.theta_mean)

    def load_weights(self, theta_mean):
        assert self.theta_mean.shape == theta_mean.shape
        self.theta_mean = theta_mean.copy()

    def load_perfect_weights(self, pr_red_ball_red_bucket, pr_red_ball_blue_bucket):
        for i in range(self.action_num):
            self.theta_mean[i*self.feature_num, i] = np.log(pr_red_ball_red_bucket/ pr_red_ball_blue_bucket)
            self.theta_mean[i*self.feature_num + 1, i] = np.log((1 - pr_red_ball_red_bucket)/ (1 - pr_red_ball_blue_bucket))
            self.theta_mean[i*self.feature_num + 2, i] = 1


    def learning_rate_decay(self, epoch, decay_rate):
        self.learning_rate_theta = 1 / (1 + decay_rate * epoch) * self.init_learning_rate_theta
        return self.learning_rate_theta

    def evaluation_init(self, pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step):
        self.pr_red_ball_red_bucket = pr_red_ball_red_bucket
        self.pr_red_ball_blue_bucket = pr_red_ball_blue_bucket
        self.evaluating = True
        self.evaluation_step = evaluation_step

    def signal_encode(self, bucket_no, ball, current_prediction):
        encoded_signal = np.zeros(shape=(1, self.feature_num * self.action_num))
        encoded_signal[0, bucket_no * self.feature_num + ball.value] = 1
        prior_index = np.arange(start=2, stop=self.feature_num * self.action_num, step=self.feature_num)

        encoded_signal[0, prior_index] = logit(current_prediction)
        return encoded_signal

    def mean_gradients_history_df(self):
        column_name_list = []
        for bucket_name in self.bucket_name_list:
            for feature_name in ['_red_ball', '_blue_ball', '_prior']:
                column_name_list.append(bucket_name + feature_name)
        mean_gradients_df_list = []
        for bucket_no in range(self.action_num):
            mean_gradients_df_list.append(
                pd.DataFrame(self.mean_gradients_history_list[bucket_no], columns=column_name_list))

        return mean_gradients_df_list

    def mean_gradients_history_plot(self):
        mean_gradients_df_list = self.mean_gradients_history_df()

        for df, bucket_no in zip(mean_gradients_df_list, range(self.action_num)):
            fig, axs = plt.subplots(self.feature_num * self.action_num,
                                    figsize=(18, 9 * self.feature_num * self.action_num))
            gradients_box_subplot(df=df.iloc[100:, :], column_list=df.columns,
                                  colour_list=['red', 'blue', 'green', 'magenta', 'indigo', 'cyan'], axs=axs)
            fig.suptitle('Bucket ' + str(bucket_no) + " Mean Gradients History")

    def mean_gradients_successive_dot_product_plot(self, moving_size=1000):
        for bucket_no in range(self.action_num):
            grad_mean_successive_dot = np.sum(
                self.mean_gradients_history_list[bucket_no] * np.roll(self.mean_gradients_history_list[bucket_no], 1,
                                                                      axis=0), axis=1)[1:]
            fig, axs = plt.subplots(2, figsize=(18, 9))
            axs[0].plot(grad_mean_successive_dot[100:], zorder=-100)
            axs[0].hlines(y=0, xmin=0, xmax=len(grad_mean_successive_dot), linestyles='dashdot', color='black',
                          zorder=-99)
            axs[0].set_title('Successive gradients dot product')
            axs[1].plot(uniform_filter1d(grad_mean_successive_dot[100:], size=moving_size), zorder=-100)
            axs[1].hlines(y=0, xmin=0, xmax=len(grad_mean_successive_dot), linestyles='dashdot', color='black',
                          zorder=-99)
            axs[1].set_title(self.name + ' Successive gradients dot product size %i moving average' % moving_size)
            fig.suptitle('Bucket ' + str(bucket_no) + ' Successive Dot Product')

    def mean_weights_history_df(self):
        column_name_list = []
        for bucket_name in self.bucket_name_list:
            for feature_name in ['_red_ball', '_blue_ball', '_prior']:
                column_name_list.append(bucket_name + feature_name)

        mean_weights_df_list = []
        for bucket_no in range(self.action_num):
            mean_weights_df_list.append(
                pd.DataFrame(self.mean_weights_history_list[bucket_no], columns=column_name_list))

        return mean_weights_df_list

    def mean_weights_history_plot(self):
        mean_weights_df_list = self.mean_weights_history_df()

        for df, title_no in zip(mean_weights_df_list, range(self.action_num)):
            fig, axs = plt.subplots(self.action_num, figsize=(18, 9 * self.action_num), squeeze=False)
            for bucket_no in range(self.action_num):
                axs[bucket_no, 0].plot(df.iloc[1:, bucket_no * self.feature_num + 0], 'r',
                                       label='Bucket ' + str(bucket_no) + ' Red weight')
                axs[bucket_no, 0].plot(df.iloc[1:, bucket_no * self.feature_num + 1], 'b',
                                       label='Bucket ' + str(bucket_no) + 'Blue weight')
                axs[bucket_no, 0].plot(df.iloc[1:, bucket_no * self.feature_num + 2], 'g',
                                       label='Bucket ' + str(bucket_no) + 'Prior weight')
                axs[bucket_no, 0].hlines(y=np.log(self.pr_red_ball_red_bucket / self.pr_red_ball_blue_bucket), xmin=0,
                                         xmax=len(df), colors='red',
                                         linestyles='dashdot')
                axs[bucket_no, 0].annotate('%.3f' % np.log(self.pr_red_ball_red_bucket / self.pr_red_ball_blue_bucket),
                                           xy=(len(df) / 2,
                                               np.log(self.pr_red_ball_red_bucket / self.pr_red_ball_blue_bucket)),
                                           xytext=(len(df) / 2, np.log(2) / 2), arrowprops=dict(arrowstyle="->"))
                axs[bucket_no, 0].hlines(
                    y=np.log((1 - self.pr_red_ball_red_bucket) / (1 - self.pr_red_ball_blue_bucket)), xmin=0,
                    xmax=len(df), colors='blue',
                    linestyles='dashdot')
                axs[bucket_no, 0].annotate(
                    '%.3f' % np.log((1 - self.pr_red_ball_red_bucket) / (1 - self.pr_red_ball_blue_bucket)),
                    xy=(len(df) / 2, np.log((1 - self.pr_red_ball_red_bucket) / (1 - self.pr_red_ball_blue_bucket))),
                    xytext=(len(df) / 2, np.log(1 / 2) / 2), arrowprops=dict(arrowstyle="->"))
                axs[bucket_no, 0].hlines(y=1, xmin=0, xmax=len(df), colors='green', linestyles='dashdot')
                axs[bucket_no, 0].hlines(y=0, xmin=0, xmax=len(df), colors='black', linestyles='dashdot')
                axs[bucket_no, 0].legend(loc='upper left')
                fig.suptitle('Bucket ' + str(title_no) + ' Mean Weights History')


class StochasticGradientAgent(Agent):

    def __init__(self, feature_num, action_num, learning_rate_theta, learning_rate_wv, memory_size=512, batch_size=16,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, learning_std=True, fixed_std=1.0, name='agent',
                 algorithm='regular'):
        # Actor weights
        super().__init__(feature_num, action_num, learning_rate_theta, name, algorithm)

        self.theta_std = np.zeros((feature_num * action_num, action_num))
        self.learning_std = learning_std
        self.fixed_std = fixed_std
        self.std_learning_rate_mask = np.ones((1, action_num))
        # Critic weights
        self.w_v = np.zeros((feature_num * action_num, action_num))
        self.learning_rate_wv = learning_rate_wv

        # Momentum variables
        self.beta1 = beta1
        self.v_dw_mean = np.zeros((feature_num * action_num, action_num))
        self.v_dw_std = np.zeros((feature_num * action_num, action_num))

        # RMSprop variables
        self.beta2 = beta2
        self.epsilon = epsilon
        self.s_dw_mean = np.zeros((feature_num * action_num, action_num))
        self.s_dw_std = np.zeros((feature_num * action_num, action_num))

        # Experience replay
        self.memory = np.zeros((memory_size, 1, (self.feature_num + 5) * self.action_num))
        self.batch_size = batch_size
        self.memory_size = memory_size

        # Evaluation
        self.std_weights_history_list = list([] for i in range(action_num))
        self.std_gradients_history_list = list([] for i in range(action_num))

        self.__print_info()

    def report(self, bucket_no, ball_colour, current_prediction, t):

        signal = self.signal_encode(bucket_no, ball_colour, current_prediction)

        ball_signal = self.signal_encode(bucket_no, ball_colour, np.ones(np.shape(current_prediction)) * 0.5)

        reported_signal_array = np.matmul(ball_signal, self.theta_mean)



        mean_array = np.matmul(signal, self.theta_mean)

        if self.learning_std:
            std_array = np.exp(np.matmul(signal, self.theta_std))
            self.std_learning_rate_mask = std_array > self.fixed_std
        else:
            std_array = np.ones((1, self.action_num)) * self.fixed_std

        h_array = np.random.normal(loc=mean_array, scale=std_array)

        # if np.any(np.isnan(h_array)):
        #     print('h_array: ', h_array)
        #     print('mean_array:', mean_array)
        #     print('std_array:', std_array)
        #     raise AssertionError('Warning: report is None !!!')

        if self.evaluating and (t % self.evaluation_step == 0):
            self.report_history_list.append([bucket_no, ball_colour.name])
            self.reward_history_list.append([bucket_no, ball_colour.name])
            for i, pr, h, mean, std in zip(range(self.action_num), current_prediction, h_array.ravel(),
                                           mean_array.ravel(), std_array.ravel()):
                self.report_history_list[-1].append(pr)
                self.report_history_list[-1].append(expit(h))
                self.report_history_list[-1].append(expit(mean))
                if bucket_no == i:
                    best_report = analytical_best_report(bucket_no, ball_colour, current_prediction,
                                                         self.pr_red_ball_red_bucket, self.pr_red_ball_blue_bucket)
                else:
                    best_report = current_prediction[i]
                self.report_history_list[-1].append(best_report)
                self.report_history_list[-1].append(std)
        return signal, h_array, mean_array, std_array, reported_signal_array

    def __print_info(self):

        print(self.name)
        print('learning_rate_theta=', self.learning_rate_theta, ' learning_rate_wv=', self.learning_rate_wv)
        if self.learning_std:
            std_string = 'learnable'
        else:
            std_string = str(self.fixed_std)
        print('memory_size=', self.memory_size, ' standard deviation=', std_string)
        print('Updating weights with ' + self.algorithm + ' algorithm.')
        print('='*30)

    def store_experience(self, t, signal_array, h_array, mean_array, std_array, reward_array):

        v_array = np.matmul(signal_array, self.w_v)
        delta = reward_array - v_array

        idx = t % self.memory_size
        self.memory[idx, 0, :self.feature_num * self.action_num] = signal_array
        self.memory[idx, 0, self.feature_num * self.action_num:(self.feature_num + 1) * self.action_num] = h_array
        self.memory[idx, 0,
        (self.feature_num + 1) * self.action_num:(self.feature_num + 2) * self.action_num] = mean_array
        self.memory[idx, 0,
        (self.feature_num + 2) * self.action_num:(self.feature_num + 3) * self.action_num] = std_array
        self.memory[idx, 0,
        (self.feature_num + 3) * self.action_num:(self.feature_num + 4) * self.action_num] = reward_array
        self.memory[idx, 0, (self.feature_num + 4) * self.action_num:(self.feature_num + 5) * self.action_num] = delta

        if self.evaluating and (t % self.evaluation_step == 0):
            for bucket_no, reward, v in zip(range(self.action_num), reward_array.ravel(), v_array.ravel()):
                self.reward_history_list[-1].append(reward)
                self.reward_history_list[-1].append(v)
                # self.reward_history_list[-1].append(compute_regret(
                #     signal_array=one_hot_decode(signal_array[:2]),
                #     pi=expit(h_array),
                #     prior_red=signal_array[2],
                #     pr_red_ball_red_bucket=self.pr_red_ball_red_bucket,
                #     pr_red_ball_blue_bucket=self.pr_red_ball_blue_bucket))
                self.mean_weights_history_list[bucket_no].append(self.theta_mean[:, bucket_no].copy().ravel())
                if self.learning_std:
                    self.std_weights_history_list[bucket_no].append(self.theta_std[:, bucket_no].copy().ravel())


    def __sample_experience(self, t):

        if t < self.batch_size:
            return self.memory[:t + 1, :, :]
        elif self.batch_size <= t < self.memory_size:
            # idx = np.random.choice(t + 1, size=self.batch_size, replace=False)
            idx = np.random.randint(low=0, high=t + 1, size=self.batch_size)  # with replacement but faster
            return self.memory[idx, :, :]
        else:
            if self.batch_size == self.memory_size:
                return self.memory
            # idx = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
            idx = np.random.randint(self.memory_size, size=self.batch_size)  # with replacement but faster
            return self.memory[idx, :, :]

    # @profile
    def batch_update(self, t):

        experience_batch = self.__sample_experience(t)

        signal_array = experience_batch[:, :, :self.feature_num * self.action_num].transpose((0, 2, 1))
        hs = experience_batch[:, :, self.feature_num * self.action_num:(self.feature_num + 1) * self.action_num]
        means = experience_batch[:, :,
                (self.feature_num + 1) * self.action_num:(self.feature_num + 2) * self.action_num]
        stds = experience_batch[:, :, (self.feature_num + 2) * self.action_num:(self.feature_num + 3) * self.action_num]
        rewards = experience_batch[:, :,
                  (self.feature_num + 3) * self.action_num:(self.feature_num + 4) * self.action_num]
        deltas = experience_batch[:, :,
                 (self.feature_num + 4) * self.action_num:(self.feature_num + 5) * self.action_num]

        batch_gradient_means = np.matmul(signal_array, deltas * ((hs - means) / np.power(stds, 2)))
        if self.learning_std:
            batch_gradient_stds = np.matmul(signal_array, deltas * (np.power(hs - means, 2) / np.power(stds, 2) - 1))
        batch_gradient_v = np.matmul(signal_array, deltas)

        gradient_mean = np.mean(batch_gradient_means, axis=0, keepdims=False)
        if self.learning_std:
            gradient_std = np.mean(batch_gradient_stds, axis=0, keepdims=False)
        gradient_v = np.mean(batch_gradient_v, axis=0, keepdims=False)

        # momentum update
        if self.algorithm == 'momentum' or self.algorithm == 'adam':
            self.v_dw_mean = self.beta1 * self.v_dw_mean + (1 - self.beta1) * gradient_mean
            if self.learning_std:
                self.v_dw_std = self.beta1 * self.v_dw_std + (1 - self.beta1) * gradient_std

        # RMSprop update
        if self.algorithm == 'adam':
            self.s_dw_mean = self.beta2 * self.s_dw_mean + (1 - self.beta2) * (np.power(gradient_mean, 2))
            if self.learning_std:
                self.s_dw_std = self.beta2 * self.s_dw_std + (1 - self.beta2) * (np.power(gradient_std, 2))

        # bias correction
        if self.algorithm == 'momentum' or self.algorithm == 'adam':
            v_dw_mean_corrected = self.v_dw_mean / (1 - np.power(self.beta1, t + 1))
            if self.learning_std:
                v_dw_std_corrected = self.v_dw_std / (1 - np.power(self.beta1, t + 1))
            if self.algorithm == 'adam':
                s_dw_mean_corrected = self.s_dw_mean / (1 - np.power(self.beta2, t + 1))
                if self.learning_std:
                    s_dw_std_corrected = self.s_dw_std / (1 - np.power(self.beta2, t + 1))

        if self.algorithm == 'momentum':
            gradient_mean = v_dw_mean_corrected
            if self.learning_std:
                gradient_std = v_dw_std_corrected
        # Adam term
        elif self.algorithm == 'adam':
            gradient_mean = (v_dw_mean_corrected / (np.sqrt(s_dw_mean_corrected) + self.epsilon))
            if self.learning_std:
                gradient_std = (v_dw_std_corrected / (np.sqrt(s_dw_std_corrected) + self.epsilon))

        # update weights

        self.theta_mean += self.learning_rate_theta * gradient_mean

        if self.learning_std:
            self.theta_std += self.std_learning_rate_mask * self.learning_rate_theta * gradient_std


        self.w_v += self.learning_rate_wv * gradient_v

        if self.evaluating and (t % self.evaluation_step == 0):
            for bucket_no in range(self.action_num):
                self.mean_gradients_history_list[bucket_no].append(gradient_mean[:, bucket_no].ravel())
                if self.learning_std:
                    self.std_gradients_history_list[bucket_no].append(gradient_std[:, bucket_no].ravel())

    def reward_history_dataframe(self):
        column_list = ['bucket_no', 'signal']
        for bucket_no in range(self.action_num):
            column_list.append('bucket_' + str(bucket_no) + '_reward')
            column_list.append('bucket_' + str(bucket_no) + '_v')
        reward_history_df = pd.DataFrame(self.reward_history_list,
                                         columns=column_list)
        return reward_history_df

    def reward_history_plot(self, top_margin=0.01, bottom_margin=0.005):
        reward_history_df = self.reward_history_dataframe()
        for bucket_no in range(self.action_num):
            fig, axs = plt.subplots(2, figsize=(15, 4 * 2))
            reward_column_name = 'bucket_' + str(bucket_no) + '_reward'
            v_column_name = 'bucket_' + str(bucket_no) + '_v'
            running_average_reward = reward_history_df[reward_column_name].expanding().mean()
            axs[0].hlines(y=0.0, xmin=0, xmax=reward_history_df.shape[0], colors='black', linestyles='dashdot')
            axs[0].plot(reward_history_df[v_column_name], label=v_column_name, zorder=-100)
            axs[0].plot(running_average_reward, zorder=-99, label='Average ' + reward_column_name)

            last_quarter_num = 3 * len(reward_history_df) // 4
            top = running_average_reward.iloc[-1] + top_margin
            bottom = running_average_reward.iloc[-1] - bottom_margin
            axs[1].plot(reward_history_df[v_column_name], zorder=-100)
            axs[1].plot(running_average_reward, zorder=-99)
            axs[1].set_xlim(left=last_quarter_num)
            axs[1].set_ylim(top=top, bottom=bottom)
            fig.legend(loc='upper right')
            fig.suptitle(self.name + ' Reward History')

    def report_history_dataframe(self):
        column_list = ['bucket_no', 'signal']
        for bucket_no in range(self.action_num):
            column_list.append('bucket_' + str(bucket_no) + '_prior')
            column_list.append('bucket_' + str(bucket_no) + '_report')
            column_list.append('bucket_' + str(bucket_no) + '_mean')
            column_list.append('bucket_' + str(bucket_no) + '_best')
            column_list.append('bucket_' + str(bucket_no) + '_std')
        report_history_df = pd.DataFrame(self.report_history_list,
                                         columns=column_list)

        return report_history_df

    def std_history_plot(self):
        if self.learning_std:
            report_history_df = self.report_history_dataframe()
            for bucket_no in range(self.action_num):
                column_name = 'bucket_' + str(bucket_no) + '_std'
                fig, ax = plt.subplots(figsize=(15, 4))
                ax.plot(report_history_df[column_name])
                ax.annotate('%.3f' % report_history_df.loc[report_history_df.index[-1], column_name],
                            xy=(
                            len(report_history_df), report_history_df.loc[report_history_df.index[-1], column_name]),
                            xytext=(len(report_history_df), 0.5), arrowprops=dict(arrowstyle="->"))
                # ax.legend(loc='lower left')
                fig.suptitle(column_name)

    # TODO: Make the report plots dependent on signal
    # TODO: Outdated. Not useful at the moment. Consider remove in the future
    def report_history_plot(self):
        report_history_df = self.report_history_dataframe()
        for bucket_no in range(self.action_num):
            fig, ax = plt.subplots(figsize=(15, 4))
            for signal, df in report_history_df.reset_index().groupby('signal_array'):
                ax.scatter(x=df['index'], y=df['report'], label=signal, marker='.', c=signal, s=3, zorder=-99)
            ax.legend(loc='lower left')
            plt.title(self.name + ' Report History')

    # TODO: Updated. Not useful at the moment.
    def mean_history_plot(self):
        report_history_df = self.report_history_dataframe()
        for bucket_no in range(2):
            fig, ax = plt.subplots(figsize=(15, 4))
            for report_bucket_no in range(2):
                mean_column_name = 'bucket_' + str(report_bucket_no) + '_mean'
                prior_column_name = 'bucket_' + str(report_bucket_no) + '_prior'
                best_column_name = 'bucket_' + str(report_bucket_no) + '_best'
                bucket_report_history_df = report_history_df[report_history_df['bucket_no'] == report_bucket_no]
                for signal, df in bucket_report_history_df.reset_index().groupby('signal'):
                    ax.scatter(x=df['index'], y=df[mean_column_name], label=signal + '_' + str(report_bucket_no),
                               marker='.', color=signal, s=0.5)

                #         red_line = mlines.Line2D([], [], color='red', label='red signal_array')
                #         blue_line = mlines.Line2D([], [], color='blue', label='blue signal_array')
                #         ax.legend(handles=[red_line, blue_line], loc='lower left')
                ax.legend(loc='lower left')
            fig.suptitle('Bucket ' + str(bucket_no) + ' Mean History')

    def std_gradients_history_df(self):
        if self.learning_std:
            column_name_list = []
            for bucket_name in self.bucket_name_list:
                for feature_name in ['_red_ball', '_blue_ball', '_prior']:
                    column_name_list.append(bucket_name + feature_name)
            std_gradients_df_list = []
            for bucket_no in range(self.action_num):
                std_gradients_df_list.append(
                    pd.DataFrame(self.std_gradients_history_list[bucket_no], columns=column_name_list))
            return std_gradients_df_list
        else:
            pass

    def std_gradients_history_plot(self):
        if self.learning_std:
            mean_gradients_df_list = self.mean_gradients_history_df()

            for df, bucket_no in zip(mean_gradients_df_list, range(self.action_num)):
                fig, axs = plt.subplots(self.feature_num * self.action_num,
                                        figsize=(18, 9 * self.feature_num * self.action_num))
                gradients_box_subplot(df=df.iloc[100:, :], column_list=df.columns,
                                      colour_list=['red', 'blue', 'green', 'magenta', 'indigo', 'cyan'], axs=axs)
                fig.suptitle('Bucket ' + str(bucket_no) + " Mean Gradients History")
        else:
            pass


class DeterministicGradientAgent(Agent):

    def __init__(self, feature_num, action_num, learning_rate_theta, learning_rate_wv,
                 learning_rate_wq, memory_size=512, batch_size=16, beta1=0.9,
                 beta2=0.999, epsilon=1e-8, name='agent', algorithm='regular'):
        # Actor weights
        super().__init__(feature_num, action_num, learning_rate_theta, name, algorithm)
        # self.theta_mean = np.zeros((feature_num * action_num, action_num))

        # Critic weights
        self.w_q = np.zeros((feature_num * action_num, action_num))
        self.w_v = np.zeros((feature_num * action_num, action_num))
        self.learning_rate_wv = learning_rate_wv
        self.learning_rate_wq = learning_rate_wq

        # Momentum variables
        self.beta1 = beta1
        self.v_dw_mean = np.zeros((feature_num * action_num, action_num))

        # RMSprop variables
        self.beta2 = beta2
        self.epsilon = epsilon
        self.s_dw_mean = np.zeros((feature_num * action_num, action_num))

        # Experience replay
        self.memory = np.zeros((memory_size, 1, (self.feature_num + 1) * self.action_num))  # signal_array, action
        self.w_q_memory = np.zeros((memory_size, feature_num * action_num, action_num))
        self.phi_memory = np.zeros((memory_size, feature_num * action_num, action_num))
        self.batch_size = batch_size
        self.memory_size = memory_size

        # Evaluation
        self.q_gradients_history_list = list([] for i in range(action_num))
        self.q_weights_history_list = list([] for i in range(action_num))
        self.v_gradients_history_list = list([] for i in range(action_num))
        self.v_weights_history_list = list([] for i in range(action_num))

        self.__print_info()

    def report(self, bucket_no, ball_colour, current_prediction, t):

        signal = self.signal_encode(bucket_no, ball_colour, current_prediction)

        mean_array = np.matmul(signal, self.theta_mean)

        # if np.any(np.isnan(mean_array)):
        #     print('mean_array:', mean_array)
        #     raise AssertionError('Warning: report is None !!!')

        if self.evaluating and (t % self.evaluation_step == 0):
            self.report_history_list.append([bucket_no, ball_colour])
            self.reward_history_list.append([bucket_no, ball_colour])
            for i, pr, mean in zip(range(self.action_num), current_prediction, mean_array.ravel()):
                self.report_history_list[-1].append(pr)
                self.report_history_list[-1].append(expit(mean))
                if bucket_no == i:
                    best_report = analytical_best_report(bucket_no, ball_colour, current_prediction,
                                                         self.pr_red_ball_red_bucket, self.pr_red_ball_blue_bucket)
                else:
                    best_report = current_prediction[i]
                self.report_history_list[-1].append(best_report)
        return signal, mean_array

    def __print_info(self):
        print(self.name)
        print('learning_rate_theta=', self.learning_rate_theta)
        print('learning_rate_wv=', self.learning_rate_wv, ' learning_rate_wq=', self.learning_rate_wq)
        print('memory_size=', self.memory_size)
        print('Updating weights with ' + self.algorithm + ' algorithm.')
        print('=' * 30)

    def store_experience(self, t, signal_array, h_array, mean_array, reward_array):

        idx = t % self.memory_size

        phi_array = np.matmul(signal_array.T, (h_array - mean_array))

        v_array = np.matmul(signal_array, self.w_v)
        q_array = np.sum(phi_array * self.w_q, axis=0, keepdims=True) + v_array

        self.memory[idx, 0, :self.feature_num * self.action_num] = signal_array
        self.memory[idx, 0, self.feature_num * self.action_num:(self.feature_num + 1) * self.action_num] = reward_array - q_array
        self.w_q_memory[idx] = self.w_q.copy()
        self.phi_memory[idx] = phi_array
        # self.memory[idx, 8:11] = self.theta_mean
        # self.memory[idx, 11:] = self.w_v

        if self.evaluating and (t % self.evaluation_step == 0):
            for bucket_no, reward, v, q in zip(range(self.action_num), reward_array.ravel(), v_array.ravel(),
                                               q_array.ravel()):
                self.reward_history_list[-1].append(reward)
                self.reward_history_list[-1].append(v)
                self.reward_history_list[-1].append(q)
                # self.reward_history_list[-1].append(compute_regret(
                #     signal_array=one_hot_decode(signal_array[:2]),
                #     pi=expit(h_array),
                #     prior_red=signal_array[2],
                #     pr_red_ball_red_bucket=self.pr_red_ball_red_bucket,
                #     pr_red_ball_blue_bucket=self.pr_red_ball_blue_bucket))
                self.mean_weights_history_list[bucket_no].append(self.theta_mean[:, bucket_no].copy().ravel())
                self.v_weights_history_list[bucket_no].append(self.w_v[:, bucket_no].copy().ravel())
                self.q_weights_history_list[bucket_no].append(self.w_q[:, bucket_no].copy().ravel())

    def __sample_experience_index(self, t):

        if t < self.batch_size:
            return np.arange(t + 1)
        elif self.batch_size <= t < self.memory_size:
            # idx = np.random.choice(t + 1, size=self.batch_size, replace=False)  # True means a value can be selected multiple times
            idx = np.random.randint(low=0, high=t + 1, size=self.batch_size)
            idx = np.arange(self.batch_size)
            return idx
        else:
            # idx = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
            idx = np.random.randint(self.memory_size, size=self.batch_size)
            idx = np.arange(self.batch_size)
            return idx

    def batch_update(self, t):

        idx = self.__sample_experience_index(t)
        experience_batch = self.memory[idx]

        signal_array = experience_batch[:, :, :self.feature_num * self.action_num].transpose((0, 2, 1))
        deltas = experience_batch[:, :, self.feature_num * self.action_num:(self.feature_num + 1) * self.action_num]
        phis = self.phi_memory[idx]
        w_qs = self.w_q_memory[idx]

        # batch_gradient_means = signals * np.sum(signals * w_qs, axis=1, keepdims=True)
        # batch_gradient_means = signals * np.dot(signals, self.w_q.T)
        batch_gradient_means = w_qs  # Natural gradient
        # batch_gradient_means = self.w_q
        batch_gradient_q = deltas * phis
        batch_gradient_v = np.matmul(signal_array, deltas)

        gradient_mean = np.mean(batch_gradient_means, axis=0, keepdims=False)
        gradient_q = np.mean(batch_gradient_q, axis=0, keepdims=False)
        gradient_v = np.mean(batch_gradient_v, axis=0, keepdims=False)

        # momentum update
        if self.algorithm == 'momentum' or self.algorithm == 'adam':
            self.v_dw_mean = self.beta1 * self.v_dw_mean + (1 - self.beta1) * gradient_mean

        # RMSprop update
        if self.algorithm == 'adam':
            self.s_dw_mean = self.beta2 * self.s_dw_mean + (1 - self.beta2) * (np.power(gradient_mean, 2))

        # bias correction
        if self.algorithm == 'momentum' or self.algorithm == 'adam':
            v_dw_mean_corrected = self.v_dw_mean / (1 - np.power(self.beta1, t + 1))
            if self.algorithm == 'adam':
                s_dw_mean_corrected = self.s_dw_mean / (1 - np.power(self.beta2, t + 1))

        if self.algorithm == 'momentum':
            gradient_mean = v_dw_mean_corrected
        # Adam term
        elif self.algorithm == 'adam':
            gradient_mean = (v_dw_mean_corrected / (np.sqrt(s_dw_mean_corrected) + self.epsilon))

        # update weights
        self.theta_mean += self.learning_rate_theta * gradient_mean

        self.w_q += self.learning_rate_wq * gradient_q
        self.w_v += self.learning_rate_wv * gradient_v

        if np.any(np.isnan(self.w_q)) or np.any(np.isnan(self.w_v)):
            print('w_q: ', self.w_q)
            print('w_v: ', self.w_v)
            raise AssertionError('Warning: weights are none !!!')

        if self.evaluating and (t % self.evaluation_step == 0):
            for bucket_no in range(self.action_num):
                self.mean_gradients_history_list[bucket_no].append(gradient_mean[:, bucket_no].ravel())
                self.v_gradients_history_list[bucket_no].append(gradient_v[:, bucket_no].ravel())
                self.q_gradients_history_list[bucket_no].append(gradient_q[:, bucket_no].ravel())

    def reward_history_dataframe(self):
        column_list = ['bucket_no', 'signal']
        for bucket_no in range(self.action_num):
            column_list.append('bucket_' + str(bucket_no) + '_reward')
            column_list.append('bucket_' + str(bucket_no) + '_v')
            column_list.append('bucket_' + str(bucket_no) + '_q')
        reward_history_df = pd.DataFrame(self.reward_history_list,
                                         columns=column_list)
        return reward_history_df

    def reward_history_plot(self):
        reward_history_df = self.reward_history_dataframe()
        for bucket_no in range(self.action_num):
            fig, axs = plt.subplots(2, figsize=(15, 4 * 2))
            reward_column_name = 'bucket_' + str(bucket_no) + '_reward'
            v_column_name = 'bucket_' + str(bucket_no) + '_v'
            q_column_name = 'bucket_' + str(bucket_no) + '_q'
            axs[0].scatter(x=reward_history_df.index, y=reward_history_df[reward_column_name], label=reward_column_name,
                           marker='.', s=3)
            axs[0].hlines(y=0.0, xmin=0, xmax=reward_history_df.shape[0], colors='black', linestyles='dashdot')
            axs[1].hlines(y=0.0, xmin=0, xmax=reward_history_df.shape[0], colors='black', linestyles='dashdot')
            axs[1].plot(reward_history_df[q_column_name], zorder=-99, label=q_column_name)
            axs[1].plot(reward_history_df[v_column_name], zorder=-98, alpha=0.8, label=v_column_name)
            # axs[1].scatter(x=reward_history_df.index, y=reward_history_df[q_column_name], label=q_column_name,
            #                marker='.', s=3, alpha=0.8, zorder=-98, color='orange')
            axs[1].plot(reward_history_df[reward_column_name].expanding().mean(), zorder=-97,
                        label='Average ' + reward_column_name)
            fig.legend(loc='upper right')
            fig.suptitle(self.name + ' Reward History')

    def mean_history_plot(self):
        report_history_df = pd.DataFrame(self.report_history_list, columns=['mean_array', 'mean_array'])
        fig, ax = plt.subplots(figsize=(15, 4))
        for signal, df in report_history_df.reset_index().groupby('signal_array'):
            ax.scatter(x=df['index'], y=expit(df['mean_array']), label=signal, marker='.', c=signal, alpha=0.6, s=0.1)
        red_line = mlines.Line2D([], [], color='red', label='red signal_array')
        blue_line = mlines.Line2D([], [], color='blue', label='blue signal_array')
        ax.legend(handles=[red_line, blue_line], loc='lower left')
        plt.title(self.name + ' Mean History')

    def gradients_history_df(self, name):
        if name == 'v':
            gradients_history_list = self.v_gradients_history_list
        elif name == 'q':
            gradients_history_list = self.q_gradients_history_list
        else:
            raise ValueError('Wrong name')
        column_name_list = []
        for bucket_name in self.bucket_name_list:
            for feature_name in ['_red_ball', '_blue_ball', '_prior']:
                column_name_list.append(bucket_name + feature_name)
        gradients_df_list = []
        for bucket_no in range(self.action_num):
            gradients_df_list.append(
                pd.DataFrame(gradients_history_list[bucket_no], columns=column_name_list))

        return gradients_df_list

    def gradients_history_plot(self, name):
        try:
            gradients_df_list = self.gradients_history_df(name)
        except ValueError:
            raise

        for df, bucket_no in zip(gradients_df_list, range(self.action_num)):
            fig, axs = plt.subplots(self.feature_num * self.action_num,
                                    figsize=(18, 9 * self.feature_num * self.action_num))
            gradients_box_subplot(df=df.iloc[100:, :], column_list=df.columns,
                                  colour_list=['red', 'blue', 'green', 'magenta', 'indigo', 'cyan'], axs=axs)
            fig.suptitle('Bucket ' + str(bucket_no) + " Mean Gradients History")

    def gradients_successive_dot_product_plot(self, name, moving_size=1000):
        if name == 'v':
            gradients_history_list = self.v_gradients_history_list
        elif name == 'q':
            gradients_history_list = self.q_gradients_history_list
        else:
            raise ValueError('Wrong name')
        for bucket_no in range(self.action_num):
            grad_v_successive_dot = np.sum(
                gradients_history_list[bucket_no] * np.roll(gradients_history_list[bucket_no], 1,
                                                            axis=0), axis=1)[1:]
            fig, axs = plt.subplots(2, figsize=(18, 9))
            axs[0].plot(grad_v_successive_dot[100:], zorder=-100)
            axs[0].hlines(y=0, xmin=0, xmax=len(grad_v_successive_dot), linestyles='dashdot', color='black',
                          zorder=-99)
            axs[0].set_title('Successive gradients dot product')
            axs[1].plot(uniform_filter1d(grad_v_successive_dot[100:], size=moving_size), zorder=-100)
            axs[1].hlines(y=0, xmin=0, xmax=len(grad_v_successive_dot), linestyles='dashdot', color='black',
                          zorder=-99)
            axs[1].set_title(self.name + ' Successive gradients dot product size %i moving average' % moving_size)
            fig.suptitle('Bucket ' + str(bucket_no) + ' Successive Dot Product')

    def weights_history_df(self, name):
        if name == 'v':
            weights_history_list = self.v_weights_history_list
        elif name == 'q':
            weights_history_list = self.q_weights_history_list
        else:
            raise ValueError('Wrong name')
        column_name_list = []
        for bucket_name in self.bucket_name_list:
            for feature_name in ['_red_ball', '_blue_ball', '_prior']:
                column_name_list.append(bucket_name + feature_name)

        weights_df_list = []
        for bucket_no in range(self.action_num):
            weights_df_list.append(
                pd.DataFrame(weights_history_list[bucket_no], columns=column_name_list))

        return weights_df_list

    def weights_history_plot(self, name):

        try:
            weights_df_list = self.weights_history_df(name)
        except ValueError:
            raise

        for df, title_no in zip(weights_df_list, range(self.action_num)):
            fig, axs = plt.subplots(self.action_num, figsize=(18, 9 * self.action_num), squeeze=False)
            for bucket_no in range(self.action_num):
                axs[bucket_no, 0].plot(uniform_filter1d(df.iloc[1:, bucket_no * self.feature_num + 0], size=1000), 'r',
                                       label='Bucket ' + str(bucket_no) + ' Red weight')
                axs[bucket_no, 0].plot(uniform_filter1d(df.iloc[1:, bucket_no * self.feature_num + 1], size=1000), 'b',
                                       label='Bucket ' + str(bucket_no) + 'Blue weight')
                axs[bucket_no, 0].plot(uniform_filter1d(df.iloc[1:, bucket_no * self.feature_num + 2], size=1000), 'g',
                                       label='Bucket ' + str(bucket_no) + 'Prior weight')
                axs[bucket_no, 0].hlines(y=np.log(self.pr_red_ball_red_bucket / self.pr_red_ball_blue_bucket), xmin=0,
                                         xmax=len(df), colors='red',
                                         linestyles='dashdot')
                axs[bucket_no, 0].annotate('%.3f' % np.log(self.pr_red_ball_red_bucket / self.pr_red_ball_blue_bucket),
                                           xy=(len(df) / 2,
                                               np.log(self.pr_red_ball_red_bucket / self.pr_red_ball_blue_bucket)),
                                           xytext=(len(df) / 2, np.log(2) / 2), arrowprops=dict(arrowstyle="->"))
                axs[bucket_no, 0].hlines(
                    y=np.log((1 - self.pr_red_ball_red_bucket) / (1 - self.pr_red_ball_blue_bucket)), xmin=0,
                    xmax=len(df), colors='blue',
                    linestyles='dashdot')
                axs[bucket_no, 0].annotate(
                    '%.3f' % np.log((1 - self.pr_red_ball_red_bucket) / (1 - self.pr_red_ball_blue_bucket)),
                    xy=(len(df) / 2, np.log((1 - self.pr_red_ball_red_bucket) / (1 - self.pr_red_ball_blue_bucket))),
                    xytext=(len(df) / 2, np.log(1 / 2) / 2), arrowprops=dict(arrowstyle="->"))
                axs[bucket_no, 0].hlines(y=1, xmin=0, xmax=len(df), colors='green', linestyles='dashdot')
                axs[bucket_no, 0].hlines(y=0, xmin=0, xmax=len(df), colors='black', linestyles='dashdot')
                axs[bucket_no, 0].legend(loc='upper left')
                fig.suptitle('Bucket ' + str(title_no) + ' Mean Weights History')
