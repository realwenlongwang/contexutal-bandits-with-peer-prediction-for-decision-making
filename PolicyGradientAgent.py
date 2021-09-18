import numpy as np
from scipy.special import logit, expit
from scipy.ndimage import uniform_filter1d
from Environment import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


class Agent:

    def __init__(self, learning_rate_theta):
        self.init_learning_rate_theta = learning_rate_theta
        self.learning_rate_theta = self.init_learning_rate_theta

    def learning_rate_decay(self, epoch, decay_rate):
        self.learning_rate_theta = 1 / (1 + decay_rate * epoch) * self.init_learning_rate_theta
        return self.learning_rate_theta


class StochasticGradientAgent(Agent):

    def __init__(self, feature_shape, learning_rate_theta, learning_rate_wv, memory_size=512, batch_size=16,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, learning_std=True, fixed_std=1.0):
        # Actor weights
        super().__init__(learning_rate_theta)
        self.theta_mean = np.zeros(feature_shape)
        self.theta_std = np.zeros(feature_shape)
        self.learning_std = learning_std
        self.fixed_std = fixed_std
        # Critic weights
        self.w_v = np.zeros(feature_shape)
        self.learning_rate_wv = learning_rate_wv

        # Momentum variables
        self.beta1 = beta1
        self.v_dw_mean = np.zeros(feature_shape)
        self.v_dw_std = np.zeros(feature_shape)

        # RMSprop variables
        self.beta2 = beta2
        self.epsilon = epsilon
        self.s_dw_mean = np.zeros(feature_shape)
        self.s_dw_std = np.zeros(feature_shape)

        # Experience replay
        self.memory = np.zeros((memory_size, feature_shape[1] + 6))
        self.batch_size = batch_size
        self.memory_size = memory_size

        # Performance record
        self.evaluating = False
        self.report_history_list = []
        self.mean_gradients_history_list = []
        self.mean_weights_history_list = []
        self.reward_history_list = []

    def evaluation_init(self, pr_red_ball_red_bucket, pr_red_ball_blue_bucket):
        self.pr_red_ball_red_bucket = pr_red_ball_red_bucket
        self.pr_red_ball_blue_bucket = pr_red_ball_blue_bucket
        self.evaluating = True


    def report(self, features_list):
        features = np.array(features_list)
        [mean] = np.dot(self.theta_mean, features.T)
        if self.learning_std:
            [std] = np.exp(np.dot(self.theta_std, features.T))
        else:
            std = self.fixed_std

        h = np.random.normal(mean, std)

        if np.isnan(h):
            print('h: ', h)
            print('mean:', mean)
            print('std:', std)
            raise AssertionError('Warning: report is None !!!')

        if self.evaluating:
            self.report_history_list.append([expit(h), mean, std, one_hot_decode(features_list[:2])])
            self.reward_history_list.append([one_hot_decode(features_list[:2]), expit(features_list[2])])

        return h, mean, std

    def __print_info(self, t, algorithm):
        if t == 0:
            print('learning_rate_theta=', self.learning_rate_theta, ' learning_rate_wv=', self.learning_rate_wv)
            if self.learning_std:
                std_string = 'learnable'
            else:
                std_string = str(self.fixed_std)
            print('memory_size=', self.memory_size, ' standard deviation=', std_string)
            print('Updating weights with ' + algorithm + ' algorithm.')

    def store_experience(self, features, h, mean, std, reward, t):

        [v] = np.dot(self.w_v, features)
        delta = reward - v

        idx = t % self.memory_size
        self.memory[idx, :3] = features
        self.memory[idx, 3] = h
        self.memory[idx, 4] = mean
        self.memory[idx, 5] = std
        self.memory[idx, 6] = reward
        self.memory[idx, 7] = delta

        if self.evaluating:
            self.reward_history_list[-1].append(reward)
            self.reward_history_list[-1].append(v)
            # self.reward_history_list[-1].append(compute_regret(
            #     signal=one_hot_decode(features[:2]),
            #     pi=expit(h),
            #     prior_red=features[2],
            #     pr_red_ball_red_bucket=self.pr_red_ball_red_bucket,
            #     pr_red_ball_blue_bucket=self.pr_red_ball_blue_bucket))
            self.mean_weights_history_list.append(self.theta_mean[0].tolist())


    def __sample_experience(self, t):

        if t < self.batch_size:
            return self.memory[:t + 1, :]
        elif self.batch_size <= t < self.memory_size:
            idx = np.random.choice(t + 1, size=self.batch_size, replace=False)
            # idx = np.random.randint(low=0, high=t + 1, size=self.batch_size) # with replacement but faster
            return self.memory[idx, :]
        else:
            idx = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
            # idx = np.random.randint(self.memory_size, size=self.batch_size) # with replacement but faster
            return self.memory[idx, :]

    def batch_update(self, t, algorithm='regular'):

        experience_batch = self.__sample_experience(t)

        signals = experience_batch[:, :3]
        hs = experience_batch[:, [3]]
        means = experience_batch[:, [4]]
        stds = experience_batch[:, [5]]
        rewards = experience_batch[:, [6]]
        deltas = experience_batch[:, [7]]

        batch_gradient_means = deltas * signals * ((hs - means) / np.power(stds, 2))
        if self.learning_std:
            batch_gradient_stds = deltas * signals * (np.power(hs - means, 2) / np.power(stds, 2) - 1)
        batch_gradient_v = deltas * signals

        gradient_mean = np.mean(batch_gradient_means, axis=0, keepdims=True)
        if self.learning_std:
            gradient_std = np.mean(batch_gradient_stds, axis=0, keepdims=True)
        gradient_v = np.mean(batch_gradient_v, axis=0, keepdims=True)

        # momentum update
        if algorithm == 'momentum' or algorithm == 'adam':
            self.v_dw_mean = self.beta1 * self.v_dw_mean + (1 - self.beta1) * gradient_mean
            if self.learning_std:
                self.v_dw_std = self.beta1 * self.v_dw_std + (1 - self.beta1) * gradient_std

        # RMSprop update
        if algorithm == 'adam':
            self.s_dw_mean = self.beta2 * self.s_dw_mean + (1 - self.beta2) * (np.power(gradient_mean, 2))
            if self.learning_std:
                self.s_dw_std = self.beta2 * self.s_dw_std + (1 - self.beta2) * (np.power(gradient_std, 2))

        # bias correction
        if algorithm == 'momentum' or algorithm == 'adam':
            v_dw_mean_corrected = self.v_dw_mean / (1 - np.power(self.beta1, t + 1))
            if self.learning_std:
                v_dw_std_corrected = self.v_dw_std / (1 - np.power(self.beta1, t + 1))
            if algorithm == 'adam':
                s_dw_mean_corrected = self.s_dw_mean / (1 - np.power(self.beta2, t + 1))
                if self.learning_std:
                    s_dw_std_corrected = self.s_dw_std / (1 - np.power(self.beta2, t + 1))

        if algorithm == 'momentum':
            gradient_mean = v_dw_mean_corrected
            if self.learning_std:
                gradient_std = v_dw_std_corrected
        # Adam term
        elif algorithm == 'adam':
            gradient_mean = (v_dw_mean_corrected / (np.sqrt(s_dw_mean_corrected) + self.epsilon))
            if self.learning_std:
                gradient_std = (v_dw_std_corrected / (np.sqrt(s_dw_std_corrected) + self.epsilon))

        # update weights

        self.theta_mean += self.learning_rate_theta * gradient_mean
        if self.learning_std:
            self.theta_std += self.learning_rate_theta * gradient_std
        self.__print_info(t, algorithm)

        self.w_v += self.learning_rate_wv * gradient_v

        if not self.learning_std:
            gradient_std = np.zeros(self.theta_std.shape)

        if self.evaluating:
            self.mean_gradients_history_list.append(gradient_mean[0, :])


    def reward_history_plot(self):
        reward_history_df = pd.DataFrame(self.reward_history_list, columns=['signal', 'prior_red', 'actual_reward', 'estimated_average_reward'])
        # fig, axs = plt.subplots(3, figsize=(15, 4*3))
        fig, axs = plt.subplots(2, figsize=(15, 4 * 2))
        axs[0].scatter(x=reward_history_df.index, y=reward_history_df['actual_reward'], label='Actual log rewards',
                       marker='.', s=3)

        # axs[2].plot(reward_history_df['regret'], 'g.',label='Regret')
        axs[0].hlines(y=0.0, xmin=0, xmax=reward_history_df.shape[0], colors='black', linestyles='dashdot')
        axs[1].hlines(y=0.0, xmin=0, xmax=reward_history_df.shape[0], colors='black', linestyles='dashdot')
        for signal, df in reward_history_df.reset_index().groupby('prior_red'):
            axs[1].plot(df['estimated_average_reward'], label='Estimated average reward ' + str(signal))
            axs[1].plot(df['actual_reward'].expanding().mean(), zorder=-99, label='Average reward ' + str(signal))
        fig.legend(loc='upper right')
        fig.suptitle('Reward Plots')

    def report_history_plot(self, prior_red_list):
        report_history_df = pd.DataFrame(self.report_history_list, columns=['report', 'mean', 'std', 'signal'])
        fig, ax = plt.subplots(figsize=(15, 4))
        for signal, df in report_history_df.reset_index().groupby('signal'):
            ax.scatter(x=df['index'], y=df['report'], label=signal, marker='.', c=signal, s=3, zorder=-99)
        for prior_red in prior_red_list:
            plt.hlines(
                y=analytical_best_report_ru_rs(
                    pr_ru=prior_red,
                    pr_rs_ru=self.pr_red_ball_red_bucket,
                    pr_rs_bu=self.pr_red_ball_blue_bucket
                ), xmin=0, xmax=len(report_history_df), colors='black', linestyles='dashdot', zorder=-98)

            plt.hlines(
                y=analytical_best_report_ru_bs(
                    pr_ru=prior_red,
                    pr_bs_ru=1 - self.pr_red_ball_red_bucket,
                    pr_bs_bu=1 - self.pr_red_ball_blue_bucket
                ), xmin=0, xmax=len(report_history_df), colors='black', linestyles='dashdot', zorder=-98)
        ax.legend(loc='lower left')
        plt.title('Report')

    def mean_gradients_history_plot(self):
        grad_mean_history_df = pd.DataFrame(self.mean_gradients_history_list,
                                            columns=['red_ball', 'blue_ball', 'prior'])
        fig, axs = plt.subplots(3, figsize=(18, 9 * 3))
        gradients_box_subplot(df=grad_mean_history_df.iloc[100:, :], column_list=grad_mean_history_df.columns,
                              colour_list=['red', 'blue', 'green'], axs=axs)
        fig.suptitle("Mean Gradients")

    def mean_gradients_successive_dot_product_plot(self, moving_size=1000):
        grad_mean_successive_dot = np.sum(
            self.mean_gradients_history_list * np.roll(self.mean_gradients_history_list, 1, axis=0), axis=1)[1:]
        fig, axs = plt.subplots(2, figsize=(14, 7))
        axs[0].plot(grad_mean_successive_dot[100:], zorder=-100)
        axs[0].hlines(y=0, xmin=0, xmax=len(grad_mean_successive_dot), linestyles='dashdot', color='black', zorder=-99)
        axs[0].set_title('Successive gradients dot product')
        axs[1].plot(uniform_filter1d(grad_mean_successive_dot[100:], size=moving_size), zorder=-100)
        axs[1].hlines(y=0, xmin=0, xmax=len(grad_mean_successive_dot), linestyles='dashdot', color='black', zorder=-99)
        axs[1].set_title('Successive gradients dot product size %i moving average' % moving_size)

    def mean_history_plot(self, prior_red_list):
        report_history_df = pd.DataFrame(self.report_history_list, columns=['report', 'mean', 'std', 'signal'])
        fig, ax = plt.subplots(figsize=(15, 4))
        for signal, df in report_history_df.reset_index().groupby('signal'):
            ax.scatter(x=df['index'], y=expit(df['mean']), label=signal, marker='.', c=signal, alpha=0.6, s=0.1)
        for prior_red in prior_red_list:
            plt.hlines(
                y=analytical_best_report_ru_rs(
                    pr_ru=prior_red,
                    pr_rs_ru=self.pr_red_ball_red_bucket,
                    pr_rs_bu=self.pr_red_ball_blue_bucket
                ), xmin=0, xmax=len(report_history_df), colors='red', linestyles='dashdot')

            plt.hlines(
                y=analytical_best_report_ru_bs(
                    pr_ru=prior_red,
                    pr_bs_ru=1 - self.pr_red_ball_red_bucket,
                    pr_bs_bu=1 - self.pr_red_ball_blue_bucket
                ), xmin=0, xmax=len(report_history_df), colors='blue', linestyles='dashdot')
        red_line = mlines.Line2D([], [], color='red', label='red signal')
        blue_line = mlines.Line2D([], [], color='blue', label='blue signal')
        # for coord in phase_change_coordinates(mark_index, mean_history_df):
        #     plt.annotate('change', xy=coord, xytext=(coord[0], 0.1), arrowprops=dict(arrowstyle="->"))
        ax.legend(handles=[red_line, blue_line], loc='lower left')
        plt.title('Mean')

    def mean_weights_history_plot(self):
        mean_weights_history_df = pd.DataFrame(self.mean_weights_history_list,
                                               columns=['red_weight', 'blue_weight', 'prior_weight'])
        fig = plt.figure(figsize=(15, 4))
        plt.plot(mean_weights_history_df.iloc[1:, 0], 'r', label='Red weight')
        plt.plot(mean_weights_history_df.iloc[1:, 1], label='Blue weight')
        plt.plot(mean_weights_history_df.iloc[1:, 2], 'g', label='Prior weight')
        plt.hlines(y=logit(self.pr_red_ball_red_bucket), xmin=0, xmax=len(mean_weights_history_df), colors='red',
                   linestyles='dashdot')
        plt.annotate('%.3f' % logit(self.pr_red_ball_red_bucket),
                     xy=(len(mean_weights_history_df) / 2, logit(self.pr_red_ball_red_bucket)),
                     xytext=(len(mean_weights_history_df) / 2, np.log(2) / 2), arrowprops=dict(arrowstyle="->"))
        plt.hlines(y=logit(self.pr_red_ball_blue_bucket), xmin=0, xmax=len(mean_weights_history_df), colors='blue',
                   linestyles='dashdot')
        plt.annotate('%.3f' % logit(self.pr_red_ball_blue_bucket),
                     xy=(len(mean_weights_history_df) / 2, logit(self.pr_red_ball_blue_bucket)),
                     xytext=(len(mean_weights_history_df) / 2, np.log(1 / 2) / 2), arrowprops=dict(arrowstyle="->"))
        plt.hlines(y=1, xmin=0, xmax=len(mean_weights_history_df), colors='green', linestyles='dashdot')
        # for coord in phase_change_coordinates(mark_index, mean_weights_history_df):
        #     plt.annotate('change', xy=coord, xytext=(coord[0], 0.1), arrowprops=dict(arrowstyle="->"))
        plt.legend()
        plt.title('Weights for Mean')

class DeterministicGradientAgent(Agent):

    def __init__(self, feature_shape, learning_rate_theta, learning_rate_wv, learning_rate_wq, memory_size=512,
                 batch_size=16,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Actor weights
        super().__init__(learning_rate_theta)
        self.theta_mean = np.zeros(feature_shape)

        # Critic weights
        self.w_q = np.zeros(feature_shape)
        self.w_v = np.zeros(feature_shape)
        self.learning_rate_wv = learning_rate_wv
        self.learning_rate_wq = learning_rate_wq

        # Momentum variables
        self.beta1 = beta1
        self.v_dw_mean = np.zeros(feature_shape)

        # RMSprop variables
        self.beta2 = beta2
        self.epsilon = epsilon
        self.s_dw_mean = np.zeros(feature_shape)

        # Experience replay
        self.memory = np.zeros((memory_size, 14))  # features, action
        self.batch_size = batch_size
        self.memory_size = memory_size

    def report(self, features_list):

        features = np.array(features_list)

        [mean] = np.dot(self.theta_mean, features)

        if np.isnan(mean):
            print('mean:', mean)
            raise AssertionError('Warning: report is None !!!')

        return mean

    def __print_info(self, t, algorithm):
        if t == 0:
            print('learning_rate_theta=', self.learning_rate_theta, ' learning_rate_wq=', self.learning_rate_wq)
            print('memory_size=', self.memory_size)
            print('Updating weights with ' + algorithm + ' algorithm.')

    def store_experience(self, features, action, reward, t):
        idx = t % self.memory_size

        theta = np.dot(features, self.theta_mean.T)
        phis = (action - theta) * features

        v = np.dot(features, self.w_v.T)
        q = np.dot(phis, self.w_q.T) + v

        self.memory[idx, :3] = features
        self.memory[idx, 3:6] = phis
        self.memory[idx, 6] = reward - q
        self.memory[idx, 7:10] = self.w_q
        # self.memory[idx, 8:11] = self.theta_mean
        # self.memory[idx, 11:] = self.w_v

        return [v, q]

    def __sample_experience(self, t):

        if t < self.batch_size:
            return self.memory[:t + 1, :]
        elif self.batch_size <= t < self.memory_size:
            idx = np.random.choice(t + 1, size=self.batch_size,
                                   replace=True)  # True means a value can be selected multiple times
            # idx = np.random.randint(low=0, high=t + 1, size=self.batch_size)
            return self.memory[idx, :]
        else:
            idx = np.random.choice(self.memory_size, size=self.batch_size, replace=True)
            # idx = np.random.randint(self.memory_size, size=self.batch_size)
            return self.memory[idx, :]

    def batch_update(self, t, algorithm='adam'):

        experience_batch = self.__sample_experience(t)

        signals = experience_batch[:, :3]
        phis = experience_batch[:, 3:6]
        deltas = experience_batch[:, [6]]
        w_qs = experience_batch[:, 7:10]

        # batch_gradient_means = signals * np.sum(signals * w_qs, axis=1, keepdims=True)
        # batch_gradient_means = signals * np.dot(signals, self.w_q.T)
        # batch_gradient_means = w_qs  # Natural gradient
        batch_gradient_means = self.w_q
        batch_gradient_q = deltas * phis
        batch_gradient_v = deltas * signals

        gradient_mean = np.mean(batch_gradient_means, axis=0, keepdims=True)
        gradient_q = np.mean(batch_gradient_q, axis=0, keepdims=True)
        gradient_v = np.mean(batch_gradient_v, axis=0, keepdims=True)

        # momentum update
        self.v_dw_mean = self.beta1 * self.v_dw_mean + (1 - self.beta1) * gradient_mean

        # RMSprop update
        self.s_dw_mean = self.beta2 * self.s_dw_mean + (1 - self.beta2) * (np.power(gradient_mean, 2))

        # bias correction
        v_dw_mean_corrected = self.v_dw_mean / (1 - np.power(self.beta1, t + 1))
        s_dw_mean_corrected = self.s_dw_mean / (1 - np.power(self.beta2, t + 1))

        # Adam term
        adam_dw_mean_corrected = (v_dw_mean_corrected / (np.sqrt(s_dw_mean_corrected) + self.epsilon))

        # update weights

        # Adam algorithm
        if algorithm == 'adam':
            self.theta_mean += self.learning_rate_theta * adam_dw_mean_corrected

        # Momentum algorithm
        elif algorithm == 'momentum':
            self.theta_mean += self.learning_rate_theta * v_dw_mean_corrected

        # Regular Update
        else:
            self.theta_mean += self.learning_rate_theta * gradient_mean

        self.__print_info(t, algorithm)
        self.w_q += self.learning_rate_wq * gradient_q
        self.w_v += self.learning_rate_wv * gradient_v

        return [gradient_mean, gradient_v, gradient_q, v_dw_mean_corrected, adam_dw_mean_corrected]
