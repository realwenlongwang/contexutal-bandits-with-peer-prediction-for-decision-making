import numpy as np
from Environment import *
from scipy.special import logit, expit
import traceback
from tqdm.notebook import tnrange
from tqdm import trange
from PolicyGradientAgent import StochasticGradientAgent, DeterministicGradientAgent
import line_profiler


def stochastic_training_notebook(agent_list, learning_rate_theta, learning_rate_wv,
                                 memory_size, batch_size, training_episodes,
                                 decay_rate, beta1, beta2, algorithm, learning_std,
                                 fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                                 prior_red_list, agent_num, action_num, score_func, decision_rule,
                                 preferred_colour_pr_list, evaluation_step):
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

    pp_agent_num = 3
    sq_agent_num = agent_num - pp_agent_num
    assert sq_agent_num >= 0

    if not agent_list:
        for i in range(sq_agent_num):
            agent = StochasticGradientAgent(feature_num=3, action_num=action_num,
                                            learning_rate_theta=learning_rate_theta, learning_rate_wv=learning_rate_wv,
                                            memory_size=memory_size, batch_size=batch_size, beta1=beta1, beta2=beta2,
                                            learning_std=learning_std, fixed_std=fixed_std, name='agent' + str(i),
                                            algorithm=algorithm)

            agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
            agent_list.append(agent)

        for i in range(pp_agent_num):
            agent = StochasticGradientAgent(feature_num=3, action_num=action_num,
                                            learning_rate_theta=learning_rate_theta, learning_rate_wv=learning_rate_wv,
                                            memory_size=memory_size, batch_size=batch_size, beta1=beta1, beta2=beta2,
                                            learning_std=learning_std, fixed_std=fixed_std, name='agent' + str(i),
                                            algorithm=algorithm)
            if i == 0 or i == 1:
                agent.load_perfect_weights(pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
            agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
            agent_list.append(agent)


    system_accumulated_reward = 0

    for t in tnrange(training_episodes):
        system_reward = stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket,
                                                    pr_red_ball_blue_bucket, agent_list, t, decay_rate, score_func,
                                                    decision_rule, preferred_colour_pr_list)
        system_accumulated_reward += system_reward

    return system_accumulated_reward


def stochastic_training(learning_rate_theta, learning_rate_wv,
                        memory_size, batch_size, training_episodes,
                        decay_rate, beta1, beta2, algorithm, learning_std,
                        fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                        prior_red_list, agent_num, action_num, score_func, decision_rule,
                        preferred_colour_pr_list, evaluation_step):
    agent_list = []

    pp_agent_num = 3
    sq_agent_num = agent_num - pp_agent_num
    assert sq_agent_num >= 0

    if not agent_list:
        for i in range(sq_agent_num):
            agent = StochasticGradientAgent(feature_num=3, action_num=action_num,
                                            learning_rate_theta=learning_rate_theta, learning_rate_wv=learning_rate_wv,
                                            memory_size=memory_size, batch_size=batch_size, beta1=beta1, beta2=beta2,
                                            learning_std=learning_std, fixed_std=fixed_std, name='agent' + str(i),
                                            algorithm=algorithm)

            agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
            agent_list.append(agent)

        for i in range(pp_agent_num):
            agent = StochasticGradientAgent(feature_num=3, action_num=action_num,
                                            learning_rate_theta=learning_rate_theta, learning_rate_wv=learning_rate_wv,
                                            memory_size=memory_size, batch_size=batch_size, beta1=beta1, beta2=beta2,
                                            learning_std=learning_std, fixed_std=fixed_std, name='agent' + str(i),
                                            algorithm=algorithm)
            if i == 0 or i == 1:
                agent.load_perfect_weights(pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
            agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
            agent_list.append(agent)

    system_accumulated_reward = 0

    for t in trange(training_episodes):
        system_reward = stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket,
                                                    pr_red_ball_blue_bucket, agent_list, t, decay_rate, score_func,
                                                    decision_rule, preferred_colour_pr_list)
        system_accumulated_reward += system_reward

    return agent_list, system_accumulated_reward


def stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                                agent_list, t, decay_rate, score_func, decision_rule, preferred_colour_pr_list):
    if prior_red_list is None:
        prior_red_instances = np.random.uniform(size=action_num)
    else:
        prior_red_instances = np.random.choice(prior_red_list, size=action_num)

    buckets = MultiBuckets(action_num, prior_red_instances, pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
    # # test normal distribution effect
    # prior_red_instances = expit(np.random.normal(loc=logit(prior_red_instances), scale=0.3))
    # pm = PredictionMarket(0, prior_red=prior_red)
    # dm = DecisionMarket(action_num, prior_red_instances, decision_rule, preferred_colour=BucketColour.RED,
    #                     preferred_colour_pr_list=preferred_colour_pr_list)
    # pp = PeerPrediction(0, np.asscalar(prior_red_instances), pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
    pd = PeerDecision(action_num, len(agent_list), prior_red_instances, pr_red_ball_red_bucket, pr_red_ball_blue_bucket)

    experience_list = []

    pp_agent_num = 3
    sq_agent_num = len(agent_list) - pp_agent_num

    for i in range(sq_agent_num):
        bucket_no, ball_colour = buckets.signal()
        signal_array, h_array, mean_array, std_array, reported_signal_array = agent_list[i].report(bucket_no, ball_colour,
                                                                            pd.read_current_pred(), t)
        pi_array = expit(h_array)
        # pi_array = np.asscalar(pi_array)
        # dm.report(pi_array)
        # pp.sq_report([pi_array, 1-pi_array], ball_enum_report)
        pd.sq_report(pi_array, reported_signal_array)
        experience_list.append([t, signal_array.copy(), h_array.copy(), mean_array.copy(), std_array.copy()])

    for i in range(sq_agent_num, len(agent_list)):
        bucket_no, ball_colour = buckets.signal()
        signal_array, h_array, mean_array, std_array, reported_signal_array = agent_list[i].report(bucket_no, ball_colour,
                                                                                      pd.read_current_pred(),
                                                                                      t)
        pi_array = expit(h_array)
        # pi_array = np.asscalar(pi_array)
        # dm.report(pi_array)
        # pp.report([pi_array, 1-pi_array], ball_enum_report)
        pd.report(pi_array, reported_signal_array)
        experience_list.append([t, signal_array.copy(), h_array.copy(), mean_array.copy(), std_array.copy()])

    # rewards_array, arm = dm.log_resolve(buckets.bucket_list)
    # rewards_array = pp.log_resolve()
    rewards_array = pd.log_resolve()

    # learning
    for agent, reward_array, experience in zip(agent_list, rewards_array, experience_list):

        experience.append(reward_array)
        agent.store_experience(*experience)
        try:
            agent.batch_update(t)
        except AssertionError:
            tb = traceback.format_exc()
            print(tb)

        agent.learning_rate_decay(epoch=t, decay_rate=decay_rate)

    # return buckets.bucket_list[arm].colour == BucketColour.RED
    return False


def deterministic_training_notebook(
        agent_list, feature_num, action_num,
        learning_rate_theta, learning_rate_wv, learning_rate_wq,
        memory_size, batch_size, training_episodes,
        decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
        pr_red_ball_blue_bucket, prior_red_list, agent_num, explorer_learning,
        fixed_std, decision_rule, preferred_colour_pr_list, evaluation_step):
    if not agent_list:
        for i in range(agent_num):
            agent = DeterministicGradientAgent(
                feature_num=feature_num, action_num=action_num,
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
            agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
            agent_list.append(agent)

    explorer = Explorer(feature_num=3, action_num=action_num, learning=explorer_learning, init_learning_rate=3e-4,
                        min_std=0.1)

    system_accumulated_reward = 0

    for t in tnrange(training_episodes):
        system_reward = deterministic_iterative_policy(
            action_num, prior_red_list, pr_red_ball_red_bucket,
            pr_red_ball_blue_bucket, agent_list, explorer,
            t, decay_rate, fixed_std, decision_rule, preferred_colour_pr_list
        )
        system_accumulated_reward += system_reward

    return system_accumulated_reward


def deterministic_training(
        feature_num, action_num,
        learning_rate_theta, learning_rate_wv, learning_rate_wq,
        memory_size, batch_size, training_episodes,
        decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
        pr_red_ball_blue_bucket, prior_red_list, agent_num, explorer_learning,
        fixed_std, decision_rule, preferred_colour_pr_list, evaluation_step):
    agent_list = []

    for i in range(agent_num):
        agent = DeterministicGradientAgent(
            feature_num=feature_num, action_num=action_num,
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
        agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
        agent_list.append(agent)

    explorer = Explorer(feature_num=3, action_num=action_num, learning=explorer_learning, init_learning_rate=3e-4,
                        min_std=0.1)

    system_accumulated_reward = 0

    for t in trange(training_episodes):
        system_reward = deterministic_iterative_policy(
            action_num, prior_red_list, pr_red_ball_red_bucket,
            pr_red_ball_blue_bucket, agent_list, explorer,
            t, decay_rate, fixed_std, decision_rule, preferred_colour_pr_list
        )
        system_accumulated_reward += system_reward


def deterministic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                                   agent_list, explorer, t, decay_rate, fixed_std, decision_rule,
                                   preferred_colour_pr_list):
    if prior_red_list is None:
        prior_red_instances = np.random.uniform(size=action_num)
    else:
        prior_red_instances = np.random.choice(prior_red_list, size=action_num)
    # Prepare a bucket and a prediction market
    buckets = MultiBuckets(action_num, prior_red_instances, pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
    dm = DecisionMarket(action_num, prior_red_instances, decision_rule, preferred_colour=BucketColour.RED,
                        preferred_colour_pr_list=preferred_colour_pr_list)

    experience_list = []

    for agent in agent_list:
        bucket_no, ball_colour = buckets.signal()
        signal_array, mean_array = agent.report(bucket_no, ball_colour, dm.read_current_pred())
        # pi = expit(mean_array)
        # actor_report = [pi, 1 - pi]
        explorer.set_parameters(mean_array=mean_array, fixed_std=fixed_std)
        e_h_array = explorer.report(signal_array)
        e_pi_array = expit(e_h_array)
        dm.report(e_pi_array)
        experience_list.append([t, signal_array, e_h_array, mean_array])

    rewards_array, arm = dm.log_resolve(buckets.bucket_list)

    # learning
    for agent, reward_array, experience in zip(agent_list, rewards_array, experience_list):

        experience.append(reward_array)
        agent.store_experience(*experience)
        # explorer.update(reward_array, mean_array)

        try:
            agent.batch_update(t)
        except AssertionError:
            tb = traceback.format_exc()
            print(tb)

        agent.learning_rate_decay(epoch=t, decay_rate=decay_rate)
        #     if explorer.learning:
        #         explorer.learning_rate_decay(epoch=t, decay_rate=0.001)

    return buckets.bucket_list[arm].colour == BucketColour.RED


if __name__ == '__main__':
    learning_rate_theta = 1e-4
    learning_rate_wv =  1e-4
    memory_size = 16
    batch_size = 16
    training_episodes = int(1e6)
    decay_rate = 0
    beta1 = 0.9
    beta2 = 0.9999
    # Algorithm: adam, momentum, regular
    algorithm = 'regular'
    learning_std = False
    fixed_std = 0.3
    # Bucket parameters
    pr_red_ball_red_bucket = 2 / 3
    pr_red_ball_blue_bucket = 1 / 3
    prior_red_list = [0.7, 0.3]
    agent_num = 5
    action_num = 1
    preferred_colour_pr_list = [0.8, 0.2]
    score_func = ScoreFunction.LOG
    decision_rule = DecisionRule.DETERMINISTIC
    evaluation_step = 1

    stochastic_training(learning_rate_theta, learning_rate_wv,
                        memory_size, batch_size, training_episodes,
                        decay_rate, beta1, beta2, algorithm, learning_std,
                        fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                        prior_red_list, agent_num, action_num, score_func, decision_rule,
                        preferred_colour_pr_list, evaluation_step)

    # feature_num = 3
    # action_num = 2
    # learning_rate_theta = 1e-4
    # decay_rate = 0  # 0.001
    # learning_rate_wv = 1e-4
    # learning_rate_wq = 1e-2
    # memory_size = 16
    # batch_size = 16
    # training_episodes = 900000
    # beta1 = 0.9
    # beta2 = 0.9999
    # fixed_std = 0.3
    # # Algorithm: adam, momentum, regular
    # algorithm = 'regular'
    # # Bucket parameters
    # prior_red_list = [3 / 4, 1 / 4]
    # pr_red_ball_red_bucket = 2 / 3
    # pr_red_ball_blue_bucket = 1 / 3
    # agent_num = 1
    #
    # explorer_learning = False
    # decision_rule = DecisionRule.STOCHASTIC
    # preferred_colour_pr_list = [0.8, 0.2]
    #
    # pp_agent_list = deterministic_training(feature_num, action_num, learning_rate_theta, learning_rate_wv, learning_rate_wq,
    #                                              memory_size, batch_size, training_episodes,
    #                                              decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
    #                                              pr_red_ball_blue_bucket, prior_red_list, agent_num,
    #                                              explorer_learning, fixed_std, decision_rule, preferred_colour_pr_list)

    # pr_ru1 = 1 / 4
    # pr_ru2 = 3 / 4
    # pr_bs_ru = 1 / 3
    # pr_bs_bu = 2 / 3
    # r1v, r2v, z = dm_expected_log_reward_blue_ball(pr_ru1, pr_ru2, pr_bs_ru, pr_bs_bu)
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # # ax.contour3D(r1v, r2v, z, 100, cmap='binary')
    # ax.plot_surface(r1v, r2v, z, rstride=1, cstride=1,
    #                 cmap='viridis', edgecolor='none')
    # ax.set_xlabel('r1')
    # ax.set_ylabel('r2')
    # ax.set_zlabel('expectation')
    # ax.view_init(90, 60)
    #
    # plt.show()
