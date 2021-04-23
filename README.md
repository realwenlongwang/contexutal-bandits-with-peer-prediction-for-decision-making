# An Implementation of Stochastic and Deterministic Gradient Policy in Continuous Bandits Problem --- Prediction Markets
---
This is a program that simulates an agent who trades in a prediction market. The problem that the prediction market aims to solve is to predict the real distribution of a random variable. We define the random variable as the colour of a bucket. The problem design comes from a human-subjective experiment for decision markets.

In this program, we implement two agents. `StochasticGradientAgent` and `DeterministicGradientAgent`. The both classes are in the file `PolicyGradientAgent.py`. The algorithm of `StochasticGradientAgent` references the "_Reinforcement Learning: An Introduction 2nd edition_" [1]. `DeterministicGradientAgent` is an implementation of "_Deterministic policy gradient algorithms_" [2]. 

> [1]Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning, Second Edition: An Introduction (Second). MIT press.

> [2] Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic policy gradient algorithms. 31st International Conference on Machine Learning, ICML 2014, 1, 605–619.

