import numpy as np



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
                 beta1=0.9, beta2=0.999, epsilon=1e-8, learning_std = True, fixed_std = 1.0):
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

        return h, mean, std

    def __print_algorithm(self, t, algorithm):
        if t == 0:
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

        return v

    def __sample_experience(self, t):

        if t < self.batch_size:
            return self.memory[:t + 1, :]
        elif self.batch_size <= t < self.memory_size:
            idx = np.random.choice(t+1, size=self.batch_size, replace=False)
            # idx = np.random.randint(low=0, high=t + 1, size=self.batch_size) # with replacement but faster
            return self.memory[idx, :]
        else:
            idx = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
            # idx = np.random.randint(self.memory_size, size=self.batch_size) # with replacement but faster
            return self.memory[idx, :]

    def batch_update(self, t, algorithm='adam'):

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
        self.v_dw_mean = self.beta1 * self.v_dw_mean + (1 - self.beta1) * gradient_mean
        if self.learning_std:
            self.v_dw_std = self.beta1 * self.v_dw_std + (1 - self.beta1) * gradient_std

        # RMSprop update
        self.s_dw_mean = self.beta2 * self.s_dw_mean + (1 - self.beta2) * (np.power(gradient_mean, 2))
        if self.learning_std:
            self.s_dw_std = self.beta2 * self.s_dw_std + (1 - self.beta2) * (np.power(gradient_std, 2))

        # bias correction
        v_dw_mean_corrected = self.v_dw_mean / (1 - np.power(self.beta1, t + 1))
        if self.learning_std:
            v_dw_std_corrected = self.v_dw_std / (1 - np.power(self.beta1, t + 1))
        s_dw_mean_corrected = self.s_dw_mean / (1 - np.power(self.beta2, t + 1))
        if self.learning_std:
            s_dw_std_corrected = self.s_dw_std / (1 - np.power(self.beta2, t + 1))

        # Adam term
        adam_dw_mean_corrected = (v_dw_mean_corrected / (np.sqrt(s_dw_mean_corrected) + self.epsilon))
        if self.learning_std:
            adam_dw_std_corrected = (v_dw_std_corrected / (np.sqrt(s_dw_std_corrected) + self.epsilon))

        # update weights

        # Adam algorithm
        if algorithm == 'adam':
            self.theta_mean += self.learning_rate_theta * adam_dw_mean_corrected
            if self.learning_std:
                self.theta_std += self.learning_rate_theta * adam_dw_std_corrected
            self.__print_algorithm(t, algorithm)

        # Momentum algorithm
        elif algorithm == 'momentum':
            self.theta_mean += self.learning_rate_theta * v_dw_mean_corrected
            if self.learning_std:
                self.theta_std += self.learning_rate_theta * v_dw_std_corrected
            self.__print_algorithm(t, algorithm)
        # Regular Update
        else:
            self.theta_mean += self.learning_rate_theta * gradient_mean
            if self.learning_std:
                self.theta_std += self.learning_rate_theta * gradient_std
            self.__print_algorithm(t, algorithm)

        self.w_v += self.learning_rate_wv * gradient_v

        if not self.learning_std:
            gradient_std = v_dw_std_corrected = adam_dw_std_corrected = np.zeros(self.theta_std.shape)

        return [gradient_mean, gradient_std, v_dw_mean_corrected, v_dw_std_corrected, adam_dw_mean_corrected,
                adam_dw_std_corrected]


class DeterministicGradientAgent(Agent):

    def __init__(self, feature_shape, learning_rate_theta, learning_rate_wq, memory_size=512, batch_size=16,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Actor weights
        super().__init__(learning_rate_theta)
        self.theta_mean = np.zeros(feature_shape)

        # Critic weights
        self.w_q = np.zeros(feature_shape)
        self.w_v = np.zeros(feature_shape)
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

    def __print_algorithm(self, t, algorithm):
        if t == 0:
            print('Updating weights with ' + algorithm + ' algorithm.')

    def store_experience(self, features, action, reward, t):
        idx = t % self.memory_size

        self.memory[idx, :3] = features
        self.memory[idx, 3] = action
        self.memory[idx, 4] = reward
        self.memory[idx, 5:8] = self.w_q
        self.memory[idx, 8:11] = self.theta_mean
        self.memory[idx, 11:] = self.w_v

    def __sample_experience(self, t):

        if t < self.batch_size:
            return self.memory[:t + 1, :]
        elif self.batch_size <= t < self.memory_size:
            idx = np.random.choice(t + 1, size=self.batch_size, replace=False)
            # idx = np.random.randint(low=0, high=t + 1, size=self.batch_size)
            return self.memory[idx, :]
        else:
            idx = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
            # idx = np.random.randint(self.memory_size, size=self.batch_size)
            return self.memory[idx, :]

    def batch_update(self, t, algorithm='adam'):

        experience_batch = self.__sample_experience(t)

        signals = experience_batch[:, :3]
        actions = experience_batch[:, [3]]
        rewards = experience_batch[:, [4]]
        w_qs = experience_batch[:, 5:8]
        theta_means = experience_batch[:, 8:11]
        w_vs = experience_batch[:, 11:]

        thetas = np.sum(signals * theta_means, axis=1, keepdims=True)
        # thetas = np.dot(signals, self.theta_mean.T)
        phis = (actions - thetas) * signals

        vs = np.sum(signals * w_vs, axis=1, keepdims=True)
        # vs = np.dot(signals, self.w_v.T)
        qs = np.sum(phis * w_qs, axis=1, keepdims=True) + vs
        # qs = np.dot(phis, self.w_q.T) + vs

        deltas = rewards - qs
        deltas2 = rewards - qs

        # batch_gradient_means = signals * np.sum(signals * w_qs, axis=1, keepdims=True)
        # batch_gradient_means = signals * np.dot(signals, self.w_q.T)
        batch_gradient_means = w_qs  # Natural gradient
        batch_gradient_q = deltas * phis
        batch_gradient_v = deltas2 * signals

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
            self.__print_algorithm(t, algorithm)

        # Momentum algorithm
        elif algorithm == 'momentum':
            self.theta_mean += self.learning_rate_theta * v_dw_mean_corrected
            self.__print_algorithm(t, algorithm)
        # Regular Update
        else:
            self.theta_mean += self.learning_rate_theta * gradient_mean
            self.__print_algorithm(t, algorithm)

        self.w_q += self.learning_rate_wq * gradient_q
        self.w_v += self.learning_rate_wq * gradient_v

        q = np.mean(qs)
        v = np.mean(vs)

        return [gradient_mean, v_dw_mean_corrected, adam_dw_mean_corrected, q, v]
