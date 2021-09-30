import numpy as np
from Environment import *
from scipy.special import logit, expit
import traceback
from tqdm.notebook import tnrange
from PolicyGradientAgent import StochasticGradientAgent, DeterministicGradientAgent


def stochastic_training_notebook(learning_rate_theta, learning_rate_wv,
                                 memory_size, batch_size, training_episodes,
                                 decay_rate, beta1, beta2, algorithm, learning_std,
                                 fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                                 prior_red_list, agent_num, action_num, score_func):
    # learning_rate_theta = 1e-4
    # learning_rate_wv = 0  # 1e-4
    # memory_size = 16
    # batch_size = 16
    # training_episodes = 900000
    # decay_rate = 0
    # beta1 = 0.9
    # beta2 = 0.9999
    # # Algorithm: adam, momentum, regular
    # algorithm = 'regular'
    # learning_std = False
    # fixed_std = 0.3
    # # Bucket parameters
    # pr_red_ball_red_bucket = 2 / 3
    # pr_red_ball_blue_bucket = 1 / 3
    # prior_red_list = [3 / 4, 1 / 4]
    # agent_num = 2
    agent_list = []

    for i in range(agent_num):
        agent = StochasticGradientAgent(feature_shape=[1, 3 * action_num], learning_rate_theta=learning_rate_theta,
                                        learning_rate_wv=learning_rate_wv,
                                        memory_size=memory_size, batch_size=batch_size,
                                        beta1=beta1, beta2=beta2, learning_std=learning_std,
                                        fixed_std=fixed_std, name='agent' + str(i), algorithm=algorithm)
        agent_list.append(agent)
        agent.evaluation_init(pr_red_ball_red_bucket=pr_red_ball_red_bucket,
                              pr_red_ball_blue_bucket=pr_red_ball_blue_bucket)

    for t in tnrange(training_episodes):
        stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket, pr_red_ball_blue_bucket, agent_list, t, decay_rate, score_func)

    return agent_list


def stochastic_training(learning_rate_theta, learning_rate_wv,
                        memory_size, batch_size, training_episodes,
                        decay_rate, beta1, beta2, algorithm, learning_std,
                        fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                        prior_red_list, agent_num, action_num, score_func):
    agent_list = []

    for i in range(agent_num):
        agent = StochasticGradientAgent(feature_shape=[1, 3 * action_num], learning_rate_theta=learning_rate_theta,
                                        learning_rate_wv=learning_rate_wv,
                                        memory_size=memory_size, batch_size=batch_size,
                                        beta1=beta1, beta2=beta2, learning_std=learning_std,
                                        fixed_std=fixed_std, name='agent' + str(i), algorithm=algorithm)
        agent.evaluation_init(pr_red_ball_red_bucket=pr_red_ball_red_bucket,
                              pr_red_ball_blue_bucket=pr_red_ball_blue_bucket)
        agent_list.append(agent)

    for t in tnrange(training_episodes):
        stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket, pr_red_ball_blue_bucket, agent_list, t, decay_rate, score_func)

    return agent_list


def stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket, pr_red_ball_blue_bucket, agent_list, t, decay_rate, score_func):
    prior_red = np.random.choice(prior_red_list)
    #     prior_red = np.random.uniform()
    buckets = MultiBuckets(action_num, prior_red, pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
    pm = PredictionMarket(0, prior_red=prior_red)
    # dm = DecisionMarket(action_num, prior_red)

    for agent in agent_list:
        bucket_no, ball = buckets.signal()
        signal = [0]*3
        signal[ball.value] = 1
        signal[2] = logit(pm.current_prediction[0])
        h, mean, std = agent.report(signal)
        pi = expit(h)
        report = [pi, 1 - pi]
        pm.report(report)
        R = pm.resolve(score_func, buckets.bucket_list[0].colour.value)

        agent.store_experience(signal, h, mean, std, R, t)
        try:
            agent.batch_update(t)
        except AssertionError:
            tb = traceback.format_exc()
            print(tb)

        agent.learning_rate_decay(epoch=t, decay_rate=decay_rate)



def deterministic_training_notebook(
        learning_rate_theta, learning_rate_wv, learning_rate_wq,
        memory_size, batch_size, training_episodes,
        decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
        pr_red_ball_blue_bucket, prior_red_list, agent_num, explorer_learning, fixed_std):
    agent_list = []

    for i in range(agent_num):
        agent = DeterministicGradientAgent(
            feature_shape=[1, 3],
            learning_rate_theta=learning_rate_theta,
            learning_rate_wv=learning_rate_wv,
            learning_rate_wq=learning_rate_wq,
            memory_size=memory_size,
            batch_size=batch_size,
            beta1=beta1,
            beta2=beta2,
            name='agent' + str(i),
            algorithm=algorithm
        )
        agent_list.append(agent)
        agent.evaluation_init(pr_red_ball_red_bucket=pr_red_ball_red_bucket,
                              pr_red_ball_blue_bucket=pr_red_ball_blue_bucket)

    explorer = Explorer(feature_shape=[1, 3], learning=explorer_learning, init_learning_rate=3e-4, min_std=0.1)

    for t in tnrange(training_episodes):
        deterministic_iterative_policy(
            prior_red_list, pr_red_ball_red_bucket,
            pr_red_ball_blue_bucket, agent_list, explorer,
            t, decay_rate, fixed_std
        )

    return agent_list

def deterministic_training(
        learning_rate_theta, learning_rate_wv, learning_rate_wq,
        memory_size, batch_size, training_episodes,
        decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
        pr_red_ball_blue_bucket, prior_red_list, agent_num, explorer_learning, fixed_std):
    agent_list = []

    for i in range(agent_num):
        agent = DeterministicGradientAgent(
            feature_shape=[1, 3],
            learning_rate_theta=learning_rate_theta,
            learning_rate_wv=learning_rate_wv,
            learning_rate_wq=learning_rate_wq,
            memory_size=memory_size,
            batch_size=batch_size,
            beta1=beta1,
            beta2=beta2,
            name='agent' + str(i),
            algorithm=algorithm
        )
        agent.evaluation_init(pr_red_ball_red_bucket=pr_red_ball_red_bucket,
                              pr_red_ball_blue_bucket=pr_red_ball_blue_bucket)
        agent_list.append(agent)

    explorer = Explorer(feature_shape=[1, 3], learning=explorer_learning, init_learning_rate=3e-4, min_std=0.1)

    for t in tnrange(training_episodes):
        deterministic_iterative_policy(
            prior_red_list, pr_red_ball_red_bucket,
            pr_red_ball_blue_bucket, agent_list, explorer,
            t, decay_rate, fixed_std)


def deterministic_iterative_policy(prior_red_list, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                                agent_list, explorer, t, decay_rate, fixed_std):
    prior_red = np.random.choice(prior_red_list)
    # Prepare a bucket and a prediction market
    bucket = Bucket(0, prior_red, pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
    pm = PredictionMarket(no=0, prior_red=prior_red)

    for agent in agent_list:
        ball = bucket.signal()
        signal = [0]*3
        signal[ball.value] = 1
        signal[2] = logit(pm.current_prediction[0])
        mean = agent.report(signal)
        # pi = expit(mean)
        # actor_report = [pi, 1 - pi]
        explorer.set_parameters(mean=mean, std=fixed_std)
        e_h = explorer.report(signal)
        e_pi = expit(e_h)
        explorer_report = [e_pi, 1 - e_pi]
        #     _pi = np.random.uniform() # uniform doesn't work, reason unknown.
        #     explorer_report = [_pi, 1-_pi]

        pm.report(explorer_report)
        R = pm.log_resolve(bucket.colour.value)

        agent.store_experience(signal, e_h, mean, R, t)
        explorer.update(R, signal)

        try:
            agent.batch_update(t)
        except AssertionError:
            tb = traceback.format_exc()
            print(tb)

        agent.learning_rate_decay(epoch=t, decay_rate=decay_rate)
        #     if explorer.learning:
        #         explorer.learning_rate_decay(epoch=t, decay_rate=0.001)


if __name__ == '__main__':
    # learning_rate_theta = 1e-4
    # learning_rate_wv = 0  # 1e-4
    # memory_size = 16
    # batch_size = 16
    # training_episodes = 300000
    # decay_rate = 0
    # beta1 = 0.9
    # beta2 = 0.9999
    # # Algorithm: adam, momentum, regular
    # algorithm = 'regular'
    # learning_std = False
    # fixed_std = 0.3
    # # Bucket parameters
    # pr_red_ball_red_bucket = 2 / 3
    # pr_red_ball_blue_bucket = 1 / 3
    # prior_red_list = [3 / 4, 1 / 4]
    # agent_num = 2
    # action_num = 1
    # score_func = ScoreFunction.LOG
    #
    # stochastic_training(learning_rate_theta, learning_rate_wv,
    #                                           memory_size, batch_size, training_episodes,
    #                                           decay_rate, beta1, beta2, algorithm, learning_std,
    #                                           fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
    #                                           prior_red_list, agent_num, action_num, score_func)


    learning_rate_theta = 1e-4
    decay_rate = 0  # 0.001
    learning_rate_wv = 1e-4
    learning_rate_wq = 1e-2
    memory_size = 16
    batch_size = 16
    training_episodes = 900000
    beta1 = 0.9
    beta2 = 0.9999
    # Algorithm: adam, momentum, regular
    algorithm = 'regular'
    # Bucket parameters
    prior_red_list = [3 / 4, 1 / 4]
    pr_red_ball_red_bucket = 2 / 3
    pr_red_ball_blue_bucket = 1 / 3
    agent_num = 1
    action_num = 1

    agent_list = deterministic_training_notebook(learning_rate_theta, learning_rate_wv, learning_rate_wq,
                                                 memory_size, batch_size, training_episodes,
                                                 decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
                                                 pr_red_ball_blue_bucket, prior_red_list, agent_num,
                                                 explorer_learning=False, fixed_std=0.3)
