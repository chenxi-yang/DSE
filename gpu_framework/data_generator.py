from args import get_args

import numpy as np
from scipy.stats import truncnorm

import benchmark
from constants import *


def dataset_arg(dataset):
    if dataset == "thermostat":
        # range_ = [55.0, 62.0]
        range_ = [55.0, 70.0]
    if dataset == "mountain_car":
        # range_ = [-0.6, -0.4]
        # range_ = [-1.2, -0.4]
        # range_ = [-0.6, 0.0]
        # range_ = [-1.6, -0.0]
        range_ = [-16.0, 0.0]
    if dataset == "unsound_1":
        range_ = [-5.0, 5.0]
    if dataset == "unsound_2_separate":
        range_ = [-5.0, 5.0]
    if dataset == "unsound_2_overall":
        range_ = [-5.0, 5.0]
    if dataset == "sampling_1":
        range_ = [-1.0, 1.0]
    
    return range_


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )


def generate_dataset(func, distribution, input_range, safe_bound, data_size=50000):
    res_list = list()
    min_tra, max_tra = 100000, -100000

    if distribution == "normal":
        l, r = input_range[0], input_range[1]
        X = get_truncated_normal((l+r)/2.0, sd=1, low=l, upp=r)
        x_list = X.rvs(data_size).tolist()
    
    max_tra_l = 0.0
    for x in x_list:
        trajectory_list = func(x, safe_bound)
        res_list.append(trajectory_list)
        max_tra_l = max(len(trajectory_list), max_tra_l)
        # min_tra, max_tra = min(trajectory_l, min_tra), max(trajectory_r, max_tra)
    
    print(f"max trajectory length: {max_tra_l}")
    # print(f"min-tra, max_tra: {min_tra}, {max_tra}")
    # thermostat: min-tra, max_tra: 52.01157295666703, 82.79782533533135
    
    return res_list


def write_dataset(res_list, path):
    f = open(path, 'w')
    f.write("trajectory_list\n")
    for trajectory_list in res_list:
        for state in trajectory_list:
            f.write(f"{state};")
        f.write(f"\n")
    f.close()
    return 


def run(safe_bound):
    args = get_args()
    dataset = args.dataset
    distribution = args.dataset_distribution

    if dataset == "thermostat":
        func = benchmark.thermostat
    if dataset == "mountain_car":
        func = benchmark.mountain_car
    if dataset == "unsound_1":
        func = benchmark.unsound_1
    if dataset == "unsound_2_separate":
        func = benchmark.unsound_2_separate
    if dataset == "unsound_2_overall":
        func = benchmark.unsound_2_overall
    if dataset == "sampling_1":
        func = benchmark.sampling_1
    
    input_range = dataset_arg(dataset)

    res_list = generate_dataset(func=func, distribution=distribution, input_range=input_range, safe_bound=safe_bound)
    write_dataset(
        res_list,
        # path=f"dataset/{dataset}_{distribution}_{input_range[0]}_{input_range[1]}_{safe_bound}.txt",
        path=f"{dataset_path_prefix}_{safe_bound}.txt"
        )


if __name__ == "__main__":
    for safe_bound in safe_range_bound_list:
        run(safe_bound)