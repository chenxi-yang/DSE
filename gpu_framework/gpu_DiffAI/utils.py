from scipy.stats import truncnorm
from scipy.stats import poisson

import numpy as np


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







