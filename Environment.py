import numpy as np
from scipy.special import expit
from scipy import stats

class PredictionMarket:
    n_outcomes = 0
    outcomes_list = []

    def __init__(self, outcomes_list):
        self.outcomes_list = outcomes_list.copy()
        self.n_outcomes = len(outcomes_list)
        self.init_prediction = np.ones(self.n_outcomes) / self.n_outcomes
        self.current_prediction = self.init_prediction

    def report(self, prediction):
        assert len(prediction) == self.n_outcomes, 'Number of outcomes not fit!'
        assert sum(prediction) == 1, print('Probabilities not sum to one!', prediction)
        # Record the contract if multiple traders.
        self.current_prediction = prediction.copy()

    def log_resolve(self, materialised_index):
        assert materialised_index < self.n_outcomes, 'Index out of scope!'
        scores = np.log(self.current_prediction) - np.log(self.init_prediction)
        return scores[materialised_index]


class Bucket:

    def __init__(self, prior_red=0.5):
        assert prior_red >= 0, 'Prior can not be negative!'
        assert prior_red <= 1, 'Prior can not greater than one!'

        self.prior_red = prior_red
        self.colour = np.random.choice(['red_bucket', 'blue_bucket'], p=(self.prior_red, 1 - self.prior_red))
        if self.colour == 'red_bucket':
            self.ball_list = ['red', 'red', 'blue']
        else:
            self.ball_list = ['blue', 'blue', 'red']

    def signal(self):
        return np.random.choice(self.ball_list)


class Explorer:
    def __init__(self, feature_shape, learning=True, init_learning_rate=0.001):
        self.mean = 0
        self.std = 1.0
        self.theta_std = np.zeros(feature_shape)
        self.init_learning_rate = init_learning_rate
        self.learning_rate = init_learning_rate
        self.h = 0
        self.learning = learning

    def set_parameters(self, mean, var=1.0):
        self.mean = mean
        self.std = var

    def learning_rate_decay(self, epoch, decay_rate):
        self.learning_rate = 1 / (1 + decay_rate * epoch) * self.init_learning_rate
        return self.learning_rate

    def report(self, x):
        x = np.array(x)
        [self.std] = np.exp(np.dot(self.theta_std, x.T))
        if self.learning:
            self.h = np.random.normal(loc=self.mean, scale=self.std)
        else:
            self.h = np.random.normal(loc=self.mean, scale=1)
        pred = expit(self.h)
        return [pred, 1 - pred]

    def update(self, reward, x):
        if self.learning:
            gradient_std = reward * np.array([x]) * ((self.h - self.mean) ** 2 / self.std ** 2 - 1)
            self.theta_std += self.learning_rate * gradient_std


def not_outlier(points, thresh=3):
    z = np.abs(stats.zscore(points))
    z = z.reshape(-1)
    return points[z < thresh]


def one_hot_encode(feature):
    if feature == 'red':
        return [1, 0]
    else:
        return [0, 1]


bucket_colour_to_num = {'red_bucket': 0, 'blue_bucket': 1}
