import sys
sys.path.append("../")

import numpy as np
from scipy.stats import truncnorm
from scipy.stats import uniform

import benchmark
from args import get_args
from constants import *


def dataset_arg(dataset):
    if benchmark_name == "thermostat":
        range_ = [55.0, 70.0]
    elif benchmark_name == "mountain_car":
        range_ = [-0.6, -0.4]
    elif benchmark_name == "unsmooth_1":
        range_ = [-5.0, 5.0]
    elif benchmark_name == "unsmooth_1_a":
        range_ = [-5.0, 5.0]
    elif benchmark_name == "unsmooth_1_b":
        range_ = [-5.0, 5.0]
    elif benchmark_name == "unsmooth_1_c":
        range_ = [-5.0, 5.0]
    elif benchmark_name == "unsmooth_2_separate":
        range_ = [-5.0, 5.0]
    elif benchmark_name == "unsmooth_2_overall":
        range_ = [-5.0, 5.0]
    elif benchmark_name == "sampling_1":
        range_ = [-1.0, 1.0]
    elif benchmark_name == "sampling_2":
        range_ = [0.0, 10.0]
    elif benchmark_name == "path_explosion":
        range_ = [2.0, 9.9]
    elif benchmark_name == "path_explosion_2":
        range_ = [2.0, 4.8]
    elif benchmark_name == "mountain_car_1":
        range_ =[-0.6, -0.4]
    elif benchmark_name == "fairness_1":
        range_ = [0.0, 10.0]
    elif benchmark_name in ["pattern6", "pattern7"]:
        range_ = [-1.0, 1.0]
    elif "pattern" in benchmark_name:
        range_ = [-5.0, 5.0]
    elif benchmark_name == "racetrack_easy":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_easy_classifier":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_easy_classifier_ITE":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_moderate_classifier_ITE":
        range_ = [7.0, 10.0]
    elif benchmark_name == "racetrack_moderate_2_classifier_ITE":
        range_ = [7.0, 10.0]
    elif benchmark_name == "racetrack_moderate_3_classifier_ITE":
        range_ = [7.0, 9.0]
    elif benchmark_name == "racetrack_hard_classifier_ITE":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_easy_1_classifier":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_easy_2_classifier":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_easy_1":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_easy_multi":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_easy_multi2":
        range_ = [4.0, 6.0]
    elif benchmark_name == "racetrack_relaxed_multi":
        range_ = [5.0, 6.0]
    elif benchmark_name == "racetrack_relaxed_multi2":
        range_ = [5.0, 6.0]
    elif benchmark_name == "racetrack_easy_sample":
        range_ = [4.0, 6.0]
    elif benchmark_name == "thermostat_refined":
        range_ = [60.0, 64.0]
    elif benchmark_name == "thermostat_new":
        range_ = [60.0, 64.0]
<<<<<<< HEAD
<<<<<<< HEAD
    elif benchmark_name == "thermostat_new_cnn":
        range_ = [60.0, 64.0]
    elif benchmark_name == "thermostat_new_tinyinput":
        range_ = [60.0, 60.1]
    elif benchmark_name == "thermostat_new_3branches":
        range_ = [60.0, 64.0]
    elif benchmark_name == "thermostat_new_40":
        range_ = [60.0, 64.0]
    elif benchmark_name == "thermostat_new_unsafe25":
        range_ = [60.0, 64.0]
    elif benchmark_name == "thermostat_new_unsafe50":
        range_ = [60.0, 64.0]
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
    elif benchmark_name == "aircraft_collision":
        range_ = [12.0, 16.0]
    elif benchmark_name == "aircraft_collision_refined":
        range_ = [12.0, 16.0]
    elif benchmark_name == "aircraft_collision_refined_classifier":
        range_ = [12.0, 16.0]
    elif benchmark_name == "aircraft_collision_refined_classifier_ITE":
        range_ = [12.0, 16.0]
    elif benchmark_name == "aircraft_collision_new":
        range_ = [12.0, 16.0]
    elif benchmark_name == "aircraft_collision_new_1":
        range_ = [12.0, 16.0]
<<<<<<< HEAD
<<<<<<< HEAD
    elif benchmark_name == "aircraft_collision_new_1_cnn":
        range_ = [12.0, 16.0]
    elif benchmark_name == "aircraft_collision_new_1_unsafe25":
        range_ = [12.0, 16.0]
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
    
    return range_


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )


<<<<<<< HEAD
<<<<<<< HEAD
def show_unsafe_percentage(res_list):
    unsafe_area_dict = {
        'thermostat': [55.0, 83.0],
        'AC': [40.0, 100000.0],
    }
    if 'thermostat' in benchmark_name:
        unsafe_area = unsafe_area_dict['thermostat']
    elif 'aircraft_collision' in benchmark_name:
        unsafe_area = unsafe_area_dict['AC']
    unsafe_num = 0.0
    for trajectory_list in res_list:
        target_list = list()
        for state in trajectory_list:
            input, output = state
            target_x = input[0]
            target_list.append(target_x)
        if max(target_list) > unsafe_area[1] or min(target_list) < unsafe_area[0]:
            unsafe_num += 1
    print(f"unsafe trajectory percentage: {unsafe_num / len(res_list)}")


=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
def generate_dataset(func, distribution, input_range, safe_bound, data_size=20000):
    res_list = list()
    min_tra, max_tra = 100000, -100000
    print(f"Start generation.")

    if distribution == "normal":
        l, r = input_range[0], input_range[1]
        X = get_truncated_normal((l+r)/2.0, sd=1, low=l, upp=r)
        x_list = X.rvs(data_size).tolist()
        x_list.append(l)
        x_list.append(r)
    
    if distribution == "uniform":
        l, r = input_range[0], input_range[1]
        x_list = (uniform.rvs(size=data_size) * (r - l) + l).tolist()
        x_list[0] = l
        x_list[1] = r
    
    max_tra_l = 0.0
    avg_tra_l = 0.0
    for x in x_list:
        trajectory_list = func(x, safe_bound)
        res_list.append(trajectory_list)
        max_tra_l = max(len(trajectory_list), max_tra_l)
        avg_tra_l = avg_tra_l + len(trajectory_list)
        # min_tra, max_tra = min(trajectory_l, min_tra), max(trajectory_r, max_tra)
    
<<<<<<< HEAD
<<<<<<< HEAD
    show_unsafe_percentage(res_list)
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
    print(f"avg trajectory length: {avg_tra_l/len(x_list)}")
    print(f"max trajectory length: {max_tra_l}")
    # print(f"min-tra, max_tra: {min_tra}, {max_tra}")
    # thermostat: min-tra, max_tra: 52.01157295666703, 82.79782533533135
    
    return res_list


def write_dataset(res_list, path):
    f = open(path, 'w')
    f.write("trajectory_list\n")
    for trajectory_list in res_list:
        if len(trajectory_list) == 0:
            continue
        for state in trajectory_list:
            f.write(f"{state};")
        f.write(f"\n")
    f.close()
    return 


def run(safe_bound):
    args = get_args()
    benchmark_name = args.benchmark_name
    distribution = args.dataset_distribution

    if benchmark_name == "thermostat":
        func = benchmark.thermostat
    elif benchmark_name == "mountain_car":
        func = benchmark.mountain_car
    elif benchmark_name == "mountain_car_1":
        func = benchmark.mountain_car_1
    elif benchmark_name == "unsmooth_1":
        func = benchmark.unsmooth_1
    elif benchmark_name == "unsmooth_1_a":
        func = benchmark.unsmooth_1_a
    elif benchmark_name == "unsmooth_1_b":
        func = benchmark.unsmooth_1_b
    elif benchmark_name == "unsmooth_1_c":
        func = benchmark.unsmooth_1_c
    elif benchmark_name == "unsmooth_2_separate":
        func = benchmark.unsmooth_2_separate
    elif benchmark_name == "unsmooth_2_overall":
        func = benchmark.unsmooth_2_overall
    elif benchmark_name == "sampling_1":
        func = benchmark.sampling_1
    elif benchmark_name == "sampling_2":
        func = benchmark.sampling_2
    elif benchmark_name == "path_explosion":
        func = benchmark.path_explosion
    elif benchmark_name == "path_explosion_2":
        func = benchmark.path_explosion_2
    elif benchmark_name == "fairness_1":
        func = benchmark.fairness_1
    elif benchmark_name == "pattern1_a":
        func = benchmark.pattern1_a
    elif benchmark_name == "pattern1_b":
        func = benchmark.pattern1_b
    elif benchmark_name == "pattern2":
        func = benchmark.pattern2
    elif benchmark_name == "pattern3_a":
        func = benchmark.pattern3_a
    elif benchmark_name == "pattern3_b":
        func = benchmark.pattern3_b
    elif benchmark_name == "pattern31_a":
        func = benchmark.pattern31_a
    elif benchmark_name == "pattern31_b":
        func = benchmark.pattern31_b
    elif benchmark_name == "pattern5_a":
        func = benchmark.pattern5_a
    elif benchmark_name == "pattern5_b":
        func = benchmark.pattern5_b
    elif benchmark_name == "pattern6":
        func = benchmark.pattern6
    elif benchmark_name == "pattern7":
        func = benchmark.pattern7
    elif benchmark_name == "pattern8":
        func = benchmark.pattern8
    elif benchmark_name == "pattern_example":
        func = benchmark.pattern_example
    elif benchmark_name == "racetrack_easy":
        func = benchmark.racetrack_easy
    elif benchmark_name == "racetrack_easy_classifier":
        func = benchmark.racetrack_easy_classifier
    elif benchmark_name == "racetrack_easy_classifier_ITE":
        func = benchmark.racetrack_easy_classifier_ITE
    elif benchmark_name == "racetrack_moderate_classifier_ITE":
        func = benchmark.racetrack_moderate_classifier_ITE
    elif benchmark_name == "racetrack_moderate_2_classifier_ITE":
        func = benchmark.racetrack_moderate_2_classifier_ITE
    elif benchmark_name == "racetrack_moderate_3_classifier_ITE":
        func = benchmark.racetrack_moderate_3_classifier_ITE
    elif benchmark_name == "racetrack_hard_classifier_ITE":
        func = benchmark.racetrack_hard_classifier_ITE
    elif benchmark_name == "racetrack_easy_1_classifier":
        func = benchmark.racetrack_easy_1_classifier
    elif benchmark_name == "racetrack_easy_2_classifier":
        func = benchmark.racetrack_easy_2_classifier
    elif benchmark_name == "racetrack_easy_1":
        func = benchmark.racetrack_easy_1
    elif benchmark_name == "racetrack_easy_multi":
        func = benchmark.racetrack_easy_multi
    elif benchmark_name == "racetrack_easy_multi2":
        func = benchmark.racetrack_easy_multi2
    elif benchmark_name == "racetrack_relaxed_multi":
        func = benchmark.racetrack_relaxed_multi
    elif benchmark_name == "racetrack_relaxed_multi2":
        func = benchmark.racetrack_relaxed_multi2
    elif benchmark_name == "racetrack_easy_sample":
        func = benchmark.racetrack_easy_sample
    elif benchmark_name == "thermostat_refined":
        func = benchmark.thermostat_refined
    elif benchmark_name == "thermostat_new":
        func = benchmark.thermostat_new
<<<<<<< HEAD
<<<<<<< HEAD
    elif benchmark_name == "thermostat_new_cnn":
        func = benchmark.thermostat_new_cnn
    elif benchmark_name == "thermostat_new_tinyinput":
        func = benchmark.thermostat_new_tinyinput
    elif benchmark_name == "thermostat_new_3branches":
        func = benchmark.thermostat_new_3branches
    elif benchmark_name == "thermostat_new_40":
        func = benchmark.thermostat_new_40
    elif benchmark_name == "thermostat_new_unsafe25":
        func = benchmark.thermostat_new_unsafe25
    elif benchmark_name == "thermostat_new_unsafe50":
        func = benchmark.thermostat_new_unsafe50
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
    elif benchmark_name == "aircraft_collision":
        func = benchmark.aircraft_collision
    elif benchmark_name == "aircraft_collision_refined":
        func = benchmark.aircraft_collision_refined
    elif benchmark_name == "aircraft_collision_refined_classifier":
        func = benchmark.aircraft_collision_refined_classifier
    elif benchmark_name == "aircraft_collision_refined_classifier_ITE":
        func = benchmark.aircraft_collision_refined_classifier_ITE
    elif benchmark_name == "aircraft_collision_new":
        func = benchmark.aircraft_collision_new
    elif benchmark_name == "aircraft_collision_new_1":
        func = benchmark.aircraft_collision_new_1
<<<<<<< HEAD
<<<<<<< HEAD
    elif benchmark_name == "aircraft_collision_new_1_cnn":
        func = benchmark.aircraft_collision_new_1_cnn
    elif benchmark_name == "aircraft_collision_new_1_unsafe25":
        func = benchmark.aircraft_collision_new_1_unsafe25
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
    # elif benchmark_name == "fairness_1":
    #     func = benchmark.fairness_1
    # elif benchmark_name == "fairness_1":
    #     func = benchmark.fairness_1
    # elif benchmark_name == "fairness_1":
    #     func = benchmark.fairness_1
    
    input_range = dataset_arg(benchmark_name)

    res_list = generate_dataset(func=func, distribution=distribution, input_range=input_range, safe_bound=safe_bound)
    write_dataset(
        res_list,
        # path=f"dataset/{dataset}_{distribution}_{input_range[0]}_{input_range[1]}_{safe_bound}.txt",
        path=f"{benchmark_name}_{safe_bound}.txt"
        )

if __name__ == "__main__":
    for safe_bound in safe_range_bound_list:
        run(safe_bound)