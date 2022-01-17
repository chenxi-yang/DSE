from args import get_args

import numpy as np
from scipy.stats import truncnorm

import benchmark


def dataset_arg(dataset):
    if dataset == "thermostat":
        range_ = [55.0, 62.0]
    if dataset == "mountain_car":
        range_ = [-0.6, -0.4]
    
    return range_


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )


def generate_dataset(func, distribution, input_range, data_size=50000):
    res_list = list()
    min_tra, max_tra = 100000, -100000

    if distribution == "normal":
        l, r = input_range[0], input_range[1]
        X = get_truncated_normal((l+r)/2.0, sd=1, low=l, upp=r)
        x_list = X.rvs(data_size).tolist()
    
    for x in x_list:
        trajectory_list = func(x)
        res_list.append(trajectory_list)
        # min_tra, max_tra = min(trajectory_l, min_tra), max(trajectory_r, max_tra)
    
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


def run():
    args = get_args()
    dataset = args.dataset
    distribution = args.dataset_distribution

    if dataset == "thermostat":
        func = benchmark.thermostat
    if dataset == "mountain_car":
        func = benchmark.mountain_car
    
    input_range = dataset_arg(dataset)

    res_list = generate_dataset(func=func, distribution=distribution, input_range=input_range)
    write_dataset(
        res_list,
        path=f"dataset/{dataset}_{distribution}_{input_range[0]}_{input_range[1]}.txt",
        )


if __name__ == "__main__":
    run()