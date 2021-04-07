from args import get_args

import numpy as np
from scipy.stats import truncnorm

import benchmark
from constants import dataset_arg


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
        x, y, trajectory_l, trajectory_r = func(x)
        res_list.append((x, y, trajectory_l, trajectory_r))
        min_tra, max_tra = min(trajectory_l, min_tra), max(trajectory_r, max_tra)
    
    print(f"min-tra, max_tra: {min_tra}, {max_tra}")
    # thermostat: min-tra, max_tra: 52.01157295666703, 82.79782533533135
    
    return res_list


def write_dataset(res_list, path):
    f = open(path, 'w')
    f.write("x, y, trajectory_l, trajectory_r\n")
    for (x, y, trajectory_l, trajectory_r) in res_list:
        f.write(f"{x}, {y}, {trajectory_l}, {trajectory_r}\n")
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