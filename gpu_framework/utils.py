from scipy.stats import truncnorm
from scipy.stats import poisson

import numpy as np
import random

np.random.seed(seed=1)
random.seed(1)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )


def generate_distribution(x, l, r, distribution, unit):
    x_list = list()
    if distribution == "uniform":
        x_list = np.random.uniform(l, r, unit).tolist()
    if distribution == "normal":
        X = get_truncated_normal(x, sd=1, low=l, upp=r)
        x_list = X.rvs(unit).tolist()
    if distribution == "beta":
        l_list = np.random.beta(2, 5, int(unit/2))
        r_list = np.random.beta(2, 5, int(unit/2))
        for v in l_list:
            x_list.append(x - v)
        for v in r_list:
            x_list.append(x + v)  
    if distribution == "original":
        x_list = [x] * unit
    
    return x_list


def ini_trajectory(trajectory):
    state = trajectory[0][0]
    action = trajectory[0][1]
    return state, action


def batch_pair(trajectory_list, data_bs=256):
    states, actions = list(), list()
    for trajectory in trajectory_list:
        for (state, action) in trajectory:
            states.append(state)
            actions.append([action])
    c = list(zip(states, actions))
    random.shuffle(c)
    states, actions = zip(*c)
    states, actions = np.array(states), np.array(actions)
    # print(states.shape, actions.shape)
    # print(f"after shuffle: {states[0], actions[0]}")
    return np.concatenate(states)[:data_bs], np.concatenate(actions)[:data_bs]
    








