import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import re
import os
import numpy as np
import pandas as pd

import seaborn as sns

from args import *
# from constants import *

import numpy as np
# import domain

# args = get_args()
# generate_all_dataset = args.generate_all_dataset
# dataset_distribution = args.dataset_distribution
# lr = args.lr
# stop_val = args.stop_val
# t_epoch = args.t_epoch
# optimizer_name = args.optimizer
# w = args.w
# benchmark_name = args.benchmark_name
# train_size = args.train_size
# test_size = args.test_size
# num_epoch = args.num_epoch
# width = args.width
# bs = args.bs
# n = args.n
# l = args.l
# nn_mode = args.nn_mode
# b = args.b
# module = args.module
# num_components = args.num_components
# verification_num_components = args.verification_num_components
# verification_num_abstract_states = args.verification_num_abstract_states
# use_smooth_kernel = args.use_smooth_kernel
# save = args.save
# test_mode = args.test_mode
# adaptive_weight = args.adaptive_weight
# outside_trajectory_loss = args.outside_trajectory_loss
# verify_outside_trajectory_loss = args.verify_outside_trajectory_loss
# # safe_start_idx = args.safe_start_idx
# # safe_end_idx = args.safe_end_idx
# # path_sample_size = args.path_sample_size
# data_attr = args.data_attr
# # thermostat: normal_55.0_62.0
# # mountain_car: normal_-0.6_-0.4
# # print(f"test_mode: {test_mode}")
# mode = args.mode
# debug = args.debug
# perturbation_width = args.perturbation_width
# real_unsafe_value =  args.real_unsafe_value
# only_data_loss = args.only_data_loss
# data_bs = args.data_bs
# fixed_dataset = args.fixed_dataset
# cuda_debug = args.cuda_debug

# sound_verify = args.sound_verify
# unsound_verify = args.unsound_verify
# assert(test_mode == (sound_verify or unsound_verify))

# # thermostat: 0.3
# # mountain_car: 0.01


# STATUS = 'Training' # a global status, if Training: use normal module, if Verifying: use sound module

# path_num_list = [50]

# K_DISJUNCTS = 10000000
# SAMPLE_SIZE = 500
# DOMAIN = "interval" # [interval, zonotope]

# CURRENT_PROGRAM = 'program' + benchmark_name # 'program_test_disjunction_2'
# DATASET_PATH = f"dataset/{benchmark_name}_{data_attr}.txt"
# MODEL_PATH = f"gpu_{mode}/models"

# # Linear nn, Sigmoid
# if benchmark_name == "thermostat":
#     x_l = [55.0]
#     x_r = [62.0]
#     # SAFE_RANGE = [55.0, 81.34] # strict
#     SAFE_RANGE = [53.0, 82.8]
#     # first expr
#     # safe_range_upper_bound_list = np.arange(82.0, 83.0, 0.1).tolist()
#     # PHI = 0.05 # unsafe probability
#     # safe_range_upper_bound_list = np.arange(82.5, 83.0, 0.15).tolist()
#     safe_range_upper_bound_list = np.arange(82.81, 82.999, 0.046).tolist()

#     PHI = 0.10
#     # SAFE_RANGE = [53.0, 82.0]
#     # SAFE_RANGE = [52.0, 83.0] # not that strict
#     # SAFE_RANGE = [50.0, 85.0] # not that loose

# if benchmark_name == "mountain_car":
#     x_l = [-0.6]
#     x_r = [-0.4]

#     # u,  p
#     safe_range_list = [[-0.8, 0.8], [0.5, 10000.0]]
#     phi_list = [0.0, 0.1]
#     if adaptive_weight:
#         w_list = [0.01, 0.99]
#     else:
#         # w_list = [0.4, 0.6]
#         w_list = [1.0, 0]
#     method_list = ['all', 'last']
#     name_list = ['acceleration', 'position']
#     # TODO: upper bound list:
#     component_bound_idx = 0
#     bound_direction_idx = 1 # left or right
#     safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()

#     # SAFE_RANGE = [100.0, 100.0]
#     # safe_range_upper_bound_list = np.arange(80.0, 96.0, 5.0).tolist()
#     # PHI = 0.1

# # if adaptive_weight:
# #     model_name_prefix = f"{benchmark_name}_{data_attr}_{n}_{lr}_{use_smooth_kernel}_{w_list}"
# # else:
# model_name_prefix = f"{benchmark_name}_{data_attr}_{n}_{lr}_{nn_mode}_{module}_{use_smooth_kernel}_{w_list}_{phi_list}"
# model_name_prefix = f"{model_name_prefix}_{outside_trajectory_loss}_{only_data_loss}_{data_bs}"
# if fixed_dataset:
#     model_name_prefix = f"{model_name_prefix}_{fixed_dataset}"

# dataset_path_prefix = f"dataset/{benchmark_name}_{dataset_distribution}_{x_l[0]}_{x_r[0]}"

# result_prefix = f"{benchmark_name}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{safe_range_list}_{safe_range_bound_list}_{phi_list}_{w_list}_{outside_trajectory_loss}_{only_data_loss}_{sound_verify}_{unsound_verify}_{data_bs}"
# if fixed_dataset:
#     result_prefix = f"{result_prefix}_{fixed_dataset}"


def read_loss(loss_path):
    q_list = list()
    c_list = list()
    with open(loss_path, 'r') as loss_f:
        loss_f.readline()
        i = 0
        for line in loss_f:
            if i % 3 == 1:
                content = line.split(",")
                q = float(content[1].split(":")[1])
                c = float(content[2].split(":")[1])
                q_list.append(q)
                c_list.append(c)
            i += 1
    return q_list, c_list


def read_loss_2(loss_path):
    q_list = list()
    c_list = list()
    with open(loss_path, 'r') as loss_f:
        # loss_f.readline()
        i = 0
        for line in loss_f:
            # print(i, line)
            if i % 2 == 0:
                content = line.split(":")
                q = float(content[1])
                q_list.append(q)
            if i % 2 == 1:
                content = line.split(",")
                c = float(content[0].split(":")[1])
                c_list.append(c)
            i += 1
    return q_list, c_list


def read_sample(sample_file):
    sample_size_list = list()
    time_list = list()
    with open(sample_file, 'r') as sample_f:
        for line in sample_f:
            content = line.split(',')
            time = float(content[0].split(':')[1])
            length = int(content[1].split(':')[1])
            sample_size_list.append(length)
            time_list.append(time)
    return sample_size_list, time_list


def read_train_log(log_file):
    q_list = list()
    c_list = list()
    with open(log_file, 'r') as log_f:
        log_f.readline()
        for line in log_f:
            # print(line)
            if 'epoch' in line and 'loss' not in line:
                content = line.split(",")
                q = float(content[1].split(":")[1])/5
                c = float(content[2].split(":")[1])/5
                # print(q, c)
                q_list.append(q)
                c_list.append(c)
                if len(q_list) >= 10:
                    break
    return q_list, c_list


def read_vary_constraint(file):
    safe_l_list, safe_r_list, p1_list, p2_list = list(), list(), list(), list()
    f = open(file, 'r')
    f.readline()
    name_line = f.readline()
    name_list = name_line[:-1].split(', ')
    name1 = name_list[-2]
    name2 = name_list[-1]
    print(name1, name2)
    for line in f:
        var = line[:-1].split(', ')
        safe_l_list.append(float(var[0]))
        safe_r_list.append(float(var[1]))
        p1_list.append(float(var[2]))
        p2_list.append(float(var[3]))
    return safe_l_list[:5], safe_r_list[:5], p1_list[:5], p2_list[:5], name1, name2


def plot_line(x_list, y_list, title, x_label, y_label, label, fig_title, c='b', log=False):
    ax = plt.subplot(111)
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    plt.plot(x_list, y_list, label=label, c=c)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if log is True:
        plt.yscale("log")

    plt.title(title)
    plt.legend()
    plt.savefig(fig_title)
    plt.close()


def plot_dot(x_list, y_list, title, x_label, y_label, label, fig_title, c='b', log=False):
    ax = plt.subplot(111)
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    plt.plot(x_list, y_list, 'o', label=label, c=c)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if log is True:
        plt.yscale("log")

    plt.title(title)
    plt.legend()
    plt.savefig(fig_title)
    plt.close()


def plot_constraint(x_list, safe_l_list, safe_r_list, p1_list, p2_list, title,  x_label, y_label, label1, label2, fig_title):
    ax = plt.subplot(111)
    # ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    
    ax2 = ax.twinx()

    lns2 = ax.plot(x_list, p1_list, marker='o', label=label1)
    lns3 = ax.plot(x_list, p2_list, marker='o', label=label2)
    ax.set_ylabel("Percentage of Safe Programs")
    ax.set_xlabel("Case Index")
    ax.grid()
    
    # ax.fill_between(x_list, safe_l_list, safe_r_list, color='C3', alpha=0.5)
    range_list = [r - safe_l_list[idx] for idx, r in enumerate(safe_r_list)]
    lns1 = ax2.plot(x_list, range_list, color='C3', label='Constraint Range')
    ax2.set_ylim([32.4,33.3])
    ax2.set_ylabel("Constraint Range")
    # ax2.legend(None)
    # ax2.grid(None)

    lns = lns2+lns3+lns1
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.xticks(range(0,5))
    
    plt.title(title)
    # plt.legend()
    # plt.grid()
    plt.savefig(fig_title)
    plt.close()


def plot_loss(loss_dir):
    for loss_file in os.listdir(loss_dir):
        loss_name = os.path.splitext(loss_file)[0]
        q_list, c_list = read_loss(loss_dir + loss_name + '.txt')
        x_list = list(range(len(q_list)))

        plot_line(x_list, q_list, title=loss_name + ' data loss', x_label='epoch', y_label='loss', label='data loss', fig_title=f"figures/loss/{loss_name}_data_loss.png", c='C0')
        plot_line(x_list, c_list, title=loss_name + ' safe loss', x_label='epoch', y_label='log(loss)', label='safe loss', fig_title=f"figures/loss/{loss_name}_safe_loss.png", c='C1')


def plot_loss_2(loss_dir):
    for loss_file in os.listdir(loss_dir):
        loss_name = os.path.splitext(loss_file)[0]
        q_list, c_list = read_loss_2(loss_dir + loss_name + '.txt')
        x_list = list(range(len(q_list)))

        plot_line(x_list, q_list, title=loss_name + ' data loss', x_label='batch', y_label='log(loss)', label='data loss', fig_title=f"figures/loss/{loss_name}_data_loss.png", c='C0', log=True)
        plot_line(x_list, c_list, title=loss_name + ' safe loss', x_label='batch', y_label='log(loss)', label='safe loss', fig_title=f"figures/loss/{loss_name}_safe_loss.png", c='C1', log=True)
    

def plot_sample(sample_file):
    sample_size_list, time_list = read_sample(sample_file)
    plot_dot(sample_size_list, time_list, title='sample time', x_label='sample size', y_label='time', label='time', fig_title=f"figures/sample/sample_time.png", c='C0')


def plot_training_loss(log_file, benchmark, method_name, log=False):
    if log:
        flag = 'log-'
    else:
        flag = ''
    q_list, c_list = read_train_log(log_file)
    x_list = list(range(len(q_list)))

    plot_line(x_list, q_list, title='training data loss', x_label='epoch', y_label=flag + 'loss', label='data loss', fig_title=f"gpu_{method_name}/figures/{benchmark}_data_loss.png", c='C0', log=log)
    plot_line(x_list, c_list, title='training safe loss', x_label='epoch', y_label=flag + 'loss', label='safe loss', fig_title=f"gpu_{method_name}/figures/{benchmark}_safe_loss.png", c='C1', log=log)


def plot_vary_constraint(file):
    safe_l_list, safe_r_list, p1, p2, name1, name2 = read_vary_constraint(file)
    x_list = list(range(len(safe_l_list)))

    plot_constraint(x_list, safe_l_list, safe_r_list, p1, p2, title='Percentage of Safe Programs with Variable Constraints ',  x_label='constraint', y_label='safe percentage', label1=name1, label2=name2, fig_title=f"figures/vary_constraint_{name1}_{name2}.png")


def plot_verification_result(result_dict, figure_name):
    # sns.set_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for method in result_dict:
        x_list = result_dict[method]['res_safe_upper_list']
        y1_list = result_dict[method]['avg_verification_probability_list']
        y2_list = result_dict[method]['safe_percentage_probability_list']
        
        sns.lineplot(x=x_list, y=y1_list, label=method, ax=ax1)
        sns.lineplot(x=x_list, y=y2_list, label=method, ax=ax2)
    
    ax1.set_xlabel('Safe Range Upper Bound')
    ax1.set_ylabel('Average Unsafe Probability of Learnt Programs')
    ax2.set_xlabel('Safe Range Upper Bound')
    ax2.set_ylabel('Percentage of Verified Safe Learnt Programs')

    plt.subplots_adjust(wspace = 0.25)
    plt.legend()
    plt.savefig(f"all_figures/{figure_name}.png")
    # plt.show()
        

def vary_safe_bound():
    args = get_args()
    lr = args.lr
    stop_val = args.stop_val
    t_epoch = args.t_epoch
    optimizer_name = args.optimizer
    w = args.w
    benchmark_name = args.benchmark_name
    train_size = args.train_size
    test_size = args.test_size
    num_epoch = args.num_epoch
    width = args.width
    bs = args.bs
    n = args.n
    l = args.l
    nn_mode = args.nn_mode
    b = args.b
    module = args.module
    num_components = args.num_components
    use_smooth_kernel = args.use_smooth_kernel
    save = args.save
    test_mode = args.test_mode
    # safe_start_idx = args.safe_start_idx
    # safe_end_idx = args.safe_end_idx
    # path_sample_size = args.path_sample_size
    data_attr = args.data_attr
    
    method_list = ['DiffAI', 'DiffAI-Kernel', 'DSE', 'SPS']

    if benchmark_name == "thermostat":
        x_l = [55.0]
        x_r = [62.0]
        # SAFE_RANGE = [55.0, 81.34] # strict
        SAFE_RANGE = [53.0, 82.8]
        # first expr
        # safe_range_upper_bound_list = np.arange(82.0, 83.0, 0.1).tolist()
        # PHI = 0.05 # unsafe probability
        safe_range_upper_bound_list = np.arange(82.5, 83.0, 0.15).tolist()
        # safe_range_upper_bound_list = np.arange(82.81, 82.999, 0.046).tolist()
        PHI = 0.10
        # SAFE_RANGE = [53.0, 82.0]
        # SAFE_RANGE = [52.0, 83.0] # not that strict
        # SAFE_RANGE = [50.0, 85.0] # not that loose

    train_time = 3
    result_dict = dict()
    for method in method_list:
        res_safe_upper_list = list()
        avg_verification_probability_list = list()
        safe_percentage_probability_list = list()
        if method == 'DiffAI':
            use_smooth_kernel = False
            log_file_name = f"thermostat_DiffAI_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}"
            log_file = f"gpu_DiffAI/result/{log_file_name}.txt"
        if method == 'DiffAI-Kernel':
            use_smooth_kernel = True
            log_file_name = f"thermostat_DiffAI_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}"
            log_file = f"gpu_DiffAI/result/{log_file_name}.txt"
        if method == 'DSE':
            use_smooth_kernel = True
            log_file_name = f"thermostat_DSE_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}"
            log_file = f"gpu_DSE/result/{log_file_name}.txt"
        if method == 'SPS':
            use_smooth_kernel = True
            log_file_name = f"thermostat_SPS_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}"
            log_file = f"gpu_SPS/result/{log_file_name}.txt"

        f = open(log_file, 'r')
        for line in f:
            if 'safe_range_upper_bound' in line:
                safe_upper_bound = float(line[:-1].split(': ')[-1])
                res_safe_upper_list.append(safe_upper_bound)
                tmp_avg_list = list()
                tmp_safe_percentage_list = list()
            if 'Details' in line:
                if len(tmp_avg_list) < 3:
                    unsafe_probability = float(line.split(',')[0].split(': ')[1])
                    tmp_avg_list.append(unsafe_probability)
                    if unsafe_probability <= PHI:
                        tmp_safe_percentage_list.append(1.0)
                    else:
                        tmp_safe_percentage_list.append(0.0)
                if len(tmp_avg_list) == 3:
                    # remove the maximum probability
                    avg_verification_probability = (sum(tmp_avg_list) - max(tmp_avg_list))/train_time
                    # keep all the probability
                    # avg_verification_probability = (sum(tmp_avg_list))/train_time
                    safe_percentage_probability = sum(tmp_safe_percentage_list) / train_time
                    avg_verification_probability_list.append(avg_verification_probability)
                    safe_percentage_probability_list.append(safe_percentage_probability)
        f.close()
        result_dict[method] = {
            'res_safe_upper_list': res_safe_upper_list,
            'avg_verification_probability_list': avg_verification_probability_list,
            'safe_percentage_probability_list': safe_percentage_probability_list,
        }

    all_result_f = open(f"all_results/thermostat_{lr}_{bs}_{num_epoch}_{train_size}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{PHI}_{safe_range_upper_bound_list}.txt", 'w')
    for method in result_dict:
        all_result_f.write(f"# {method}\n")
        for key in result_dict[method]:
            all_result_f.write(f"{key}: {result_dict[method][key]}\n")
    all_result_f.close()

    plot_verification_result(result_dict, figure_name=f"thermostat_{lr}_{bs}_{num_epoch}_{train_size}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{PHI}_{safe_range_upper_bound_list}")


def plot_component(x_list, y_list_list, label_name_list, fig_name):
    for idx, y_list in enumerate(y_list_list):
        sns.lineplot(x=x_list, y=y_list, label=label_name_list[idx])

    plt.xlabel('Number of Components')
    plt.ylabel('Probability Upper Bound')
    plt.legend()
    plt.grid()
    plt.savefig(f"all_figures/{fig_name}.png")
    plt.close()


def abstract_component():
    import constants

    if mode == 'DSE':
        from gpu_DSE.data_generator import load_data
    if mode == 'DiffAI':
        from gpu_DiffAI.data_generator import load_data
    if mode == 'SPS':
        # from gpu_SPS.train import extract_abstract_representation
        from gpu_SPS.data_generator import load_data
    if mode == 'SPS-sound':
        # from gpu_SPS_sound.train import extract_abstract_representation
        from gpu_SPS_sound.data_generator import load_data

    from evaluation import verification
    import domain

    import random
    import time

    from utils import (
        extract_abstract_representation,
    )

    Trajectory_train, Trajectory_test = load_data(train_size=train_size, test_size=test_size, dataset_path=DATASET_PATH)
    num_component_list = np.arange(1, 501, 10).tolist()
    sum_list = list()
    avg_list = list()
    min_list = list()
    max_list = list()
    for num_components in num_component_list:
        component_list = extract_abstract_representation(Trajectory_train, x_l, x_r, num_components, w=perturbation_width)
        component_p_list = list()
        for component in component_list:
            component_p_list.append(component['p'])
        sum_list.append(sum(component_p_list))
        avg_list.append(sum(component_p_list) / len(component_p_list))
        min_list.append(min(component_p_list))
        max_list.append(max(component_p_list))
    
    plot_component(num_component_list, [sum_list], ['sum'], fig_name=f"{benchmark_name}_component_probability_upper_bound_sum")
    plot_component(num_component_list, [min_list, avg_list, max_list], ['min', 'avg', 'max'], fig_name=f"{benchmark_name}_component_probability_upper_bound_distribution")

    # sns.set_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for method in result_dict:
        x_list = result_dict[method]['res_safe_upper_list']
        y1_list = result_dict[method]['avg_verification_probability_list']
        y2_list = result_dict[method]['safe_percentage_probability_list']
        
        sns.lineplot(x=x_list, y=y1_list, label=method, ax=ax1)
        sns.lineplot(x=x_list, y=y2_list, label=method, ax=ax2)
    
    ax1.set_xlabel('Safe Range Upper Bound')
    ax1.set_ylabel('Average Unsafe Probability of Learnt Programs')
    ax2.set_xlabel('Safe Range Upper Bound')
    ax2.set_ylabel('Percentage of Verified Safe Learnt Programs')

    plt.subplots_adjust(wspace = 0.25)
    plt.legend()
    plt.savefig(f"all_figures/{figure_name}.png")


def plot_verify(verify_result, fig_name):
    fig, ax1 = plt.subplots(1, 1)
    for verify_dict in verify_result:
        name = verify_dict["name"]
        range_bound_list = verify_dict["range_bound_list"]
        unsafe_probability_list = verify_dict["unsafe_probability_list"]
        # print(len(range_bound_list),  len(unsafe_probability_list))
        # if 
        sns.lineplot(x=range_bound_list[:len(unsafe_probability_list)], y=unsafe_probability_list, label=name, ax=ax1, marker='o')
    
    ax1.set_xlabel("Safe Range Bound")
    ax1.set_ylabel("Unsafe Probability of Learnt Programs")

    plt.axhline(y=0.1, color='r', linestyle='-')
    plt.legend()
    plt.grid()
    plt.savefig(f"all_figures/{fig_name}.png")


def read_verify_log(file_name):
    f = open(file_name, 'r')
    f.readline()
    range_bound_list, unsafe_probability_list = list(), list()
    for line in f:
        # print(line)
        if 'safa_range_bound' in line:
            # print(line)
            safe_upper_bound = float(line[:-1].split(': ')[-1])
            range_bound_list.append(safe_upper_bound)
        if 'Details' in line:
            unsafe_probability = float(line.split(',')[0].split(': ')[1])
            unsafe_probability_list.append(unsafe_probability)
    # print(range_bound_list, unsafe_probability_list)
    return range_bound_list, unsafe_probability_list


def extract_verify_result():
    verify_list = [(10, 1), (50, 1), (50, 2), (500, 1)]
    verify_result = list()
    for num_components, num_abstract_states in verify_list:
        tmp_result_prefix = f"{result_prefix}_{num_components}_{num_abstract_states}_{verify_outside_trajectory_loss}"
        file_name = f"gpu_{mode}/result_test/{tmp_result_prefix}_evaluation.txt"
        range_bound_list, unsafe_probability_list = read_verify_log(file_name)
        verify_dict = dict(
            name=f"C({num_components}), AS({num_abstract_states})",
            range_bound_list=range_bound_list,
            unsafe_probability_list=unsafe_probability_list,
        )
        verify_result.append(verify_dict)
    plot_verify(verify_result, fig_name=f"{result_prefix}")


def plot_line(
    x_list, 
    y_list_list, 
    label_name_list, 
    figure_name, 
    x_label, 
    y_label, 
    constraint=None,
    figure_save_name=None,
    y_log=False
    ):
    color_list = ['g', 'b', 'r', 'y']

    patch_list = list()
    for idx, label in enumerate(label_name_list):
        patch_list.append(mpatches.Patch(color=color_list[idx], label=label_name_list[idx]))

    for idx, y_list in enumerate(y_list_list):
        if len(y_list) < len(x_list):
            y_list.extend([np.nan]*(len(x_list) - len(y_list)))
        sns.pointplot(x=x_list, y=y_list, marker='o', color=color_list[idx])
    
    if constraint is not None:
        plt.axhline(y=constraint, color='r', linestyle='-')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_log:
        plt.yscale('log')
    plt.title(figure_name)
    plt.legend(handles=patch_list)
    plt.grid()
    if figure_save_name is None:
        figure_save_name = figure_name
    plt.savefig(f"all_figures/{figure_save_name}.png")
    plt.close()


def provable_safe(
        benchmark_name,
        mode,
        p_list,
    ):
    for unsafe_p in p_list:
        res_file = open(f"all_results/{benchmark_name}_{mode}_{unsafe_p}.txt", 'r')
        res_file.readline()
        label_name_list = res_file.readline()[:-1].split(', ')
        x_list, y1_list, y2_list = list(), list(), list()
        for line in res_file:
            data = line[:-1].split(',')
            print(data)
            x, y1, y2 = float(data[0]), float(data[1]), float(data[2])
            x_list.append(x)
            y1_list.append(y1)
            y2_list.append(y2)
        plot_line(
            x_list,
            [y1_list, y2_list], 
            label_name_list, 
            f"{benchmark_name}_{mode}_{unsafe_p}",
            x_label=f"Safe Bound",
            y_label=f"Verified Probability of the Learnt Program",
            constraint=unsafe_p, 
            )

    return


def empirical_safe(
        benchmark_name,
        mode,
        p_list,
    ):
    for unsafe_p in p_list:
        res_file = open(f"all_results/{benchmark_name}_{mode}_{unsafe_p}.txt", 'r')
        res_file.readline()
        label_name_list = res_file.readline()[:-1].split(', ')
        x_list, y1_list, y2_list = list(), list(), list()
        for line in res_file:
            data = line[:-1].split(',')
            x, y1, y2 = float(data[0]), float(data[1]), float(data[2])
            x_list.append(x)
            y1_list.append(np.nan if y1 < 0 else y1)
            y2_list.append(np.nan if y2 < 0 else y2)
        plot_line(
            x_list,
            [y1_list, y2_list], 
            label_name_list, 
            f"{benchmark_name}_{mode}_{unsafe_p}",
            x_label=f"Safe Bound",
            y_label=f"Test Data Loss",
            )
    return 


def plot_long_line(x_list, y_list_list, label_name_list, figure_name, x_label, y_label, constraint=None):
    color_list = ['g', 'b']
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, y_list in enumerate(y_list_list):
        sns.lineplot(ax=ax, x=x_list, y=y_list, label=label_name_list[idx], color=color_list[idx])
    
    if constraint is not None:
        plt.axhline(y=constraint, color='r', linestyle='-')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(figure_name)
    # plt.grid()
    plt.savefig(f"all_figures/trajectories/{figure_name}.png")
    plt.close()


def extract_trajectory(
        mode,
        file_name,
        name_list,
        name_idx,
    ):
    f = open(f"gpu_{mode}/result_test/trajectory/{file_name}.txt", 'r')
    f.readline()
    f.readline()
    left_list, right_list = list(), list()
    for line in f:
        if 'symbol_table' in line:
            if len(left_list) > 0:
                plot_long_line(
                    x_list, 
                    [left_list, right_list],
                    ['lower bound', 'upper bound'],
                    f"{mode}_{name_list[name_idx]}_{symbol_table_idx}",
                    x_label='Iteration',
                    y_label=f"{name_list[name_idx]}",
                    )
            symbol_table_idx = float(line[:-1].split(' ')[1])
            left_list, right_list = list(), list()
            continue
        data = line[:-1].split(';')
        state = data[name_idx]
        state_data = state.split(', ')
        left, right = float(state_data[0]), float(state_data[1])
        left_list.append(left)
        right_list.append(right)
        x_list = [i for i in range(len(left_list))]

    return 


def extract_test_info(
    file_name_list,
    method_name_list,
    benchmark_name,
    ):
    # unsafe_p_file_name = f"all_results/{benchmark_name}_unsafe_p.txt"
    # data_loss_file_name = f"all_results/{benchmark_name}_data_loss.txt"
    # unsafe_p_f = open(unsafe_p_file_name, 'w')
    # data_loss_f = open(data_loss_file_name, 'w')
    method_res_dict = dict()
    all_safe_bound_list = list()
    all_data_loss_list = list()
    all_unsafe_p_list = list()

    get_safe_bound = False
    for idx, file_name in enumerate(file_name_list):
        f = open(file_name, 'r')
        f.readline()
        f.readline()
        safe_bound_list = list()
        data_loss_list = list()
        unsafe_p_list = list()
        
        tmp_data_loss_list = list()
        tmp_unsafe_p_list = list()
        for line in f:
            if 'range_bound' in line:
                if len(tmp_data_loss_list) > 0:
                    data_loss_list.append(sum(tmp_data_loss_list)/len(tmp_data_loss_list))
                    unsafe_p_list.append(sum(tmp_unsafe_p_list)/len(tmp_unsafe_p_list))
                safe_range_bound = float(line[:-1].split(': ')[-1])
                # print(line)
                # print(safe_range_bound)
                safe_bound_list.append(safe_range_bound)
                tmp_data_loss_list = list()
                tmp_unsafe_p_list = list()
            if 'Details' in line:
                unsafe_p = float(line.split(', ')[0].split(': ')[1])
                # print(unsafe_p)
                tmp_unsafe_p_list.append(unsafe_p)
            if 'test data loss' in line:
                data_loss = float(line[:-1].split(': ')[1])
                if benchmark_name == "Thermostat":
                    data_loss = data_loss ** 0.5
                # print(data_loss)
                tmp_data_loss_list.append(data_loss)

        if len(tmp_data_loss_list) > 0:
            data_loss_list.append(sum(tmp_data_loss_list)/len(tmp_data_loss_list))
            unsafe_p_list.append(sum(tmp_unsafe_p_list)/len(tmp_unsafe_p_list))
        
        method_name = method_name_list[idx]
        method_res_dict[method_name] = {
            'safe_bound_list': safe_bound_list,
            'data_loss_list': data_loss_list,
            'unsafe_p_list': unsafe_p_list,
        }
        print(f"method: {method_name_list[idx]}")
        print(safe_bound_list)
        print(data_loss_list)
        print(unsafe_p_list)
        all_safe_bound_list = safe_bound_list[:]
        all_data_loss_list.append(data_loss_list[:])
        all_unsafe_p_list.append(unsafe_p_list[:])

    # all_safe_bound_list = method_res[method_name_list[0]]['safe_bound_list']
    # all_data_loss_list = [method_res_dict[method_name]['data_loss_list'] for method_name in method_res_dict]
    # all_unsafe_p_list = [method_res_dict[method_name]['unsafe_p_list'] for method_name in method_res_dict]

    plot_line(
        x_list = all_safe_bound_list, 
        y_list_list = all_unsafe_p_list, 
        label_name_list = method_name_list, 
        figure_name=f"{benchmark_name} Verified Unsafe Probability", 
        x_label="Safe Bound", 
        y_label="Verified Unsafe Probability", 
        figure_save_name=f"{benchmark_name}_unsafe_p",
        y_log=True,
        constraint=None,
        )
        
    plot_line(
        x_list = all_safe_bound_list, 
        y_list_list = all_data_loss_list, 
        label_name_list = method_name_list, 
        figure_name=f"{benchmark_name} Data Loss", 
        x_label="Safe Bound", 
        y_label="Data Loss", 
        figure_save_name=f"{benchmark_name}_data_loss",
        constraint=None,
        )

    return 


# def plot_bar(
#     x_list,
#     y_list,
#     x_label,
#     y_label,
#     figure_name,
#     figure_save_name=None,
#     y_log=None,
#     ):
#     color_list = ['g', 'b', 'r', 'y']

#     # patch_list = list()
#     # for idx, label in enumerate(label_name_list):
#     #     patch_list.append(mpatches.Patch(color=color_list[idx], label=label_name_list[idx]))

#     sns.barplot(x=x_list, y=y_list, )
#     for idx, y_list in enumerate(y_list_list):
#         if len(y_list) < len(x_list):
#             y_list.extend([np.nan]*(len(x_list) - len(y_list)))
#         sns.pointplot(x=x_list, y=y_list, marker='o', color=color_list[idx])
    
#     if constraint is not None:
#         plt.axhline(y=constraint, color='r', linestyle='-')

#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     if y_log:
#         plt.yscale('log')
#     plt.title(figure_name)
#     plt.legend(handles=patch_list)
#     plt.grid()
#     if figure_save_name is None:
#         figure_save_name = figure_name
#     plt.savefig(f"all_figures/{figure_save_name}.png")
#     plt.close()
#     return 
            

def extract_running_time(
        file_name='all_results/thermostat_running_time.txt',
        benchmark_name='Thermostat',
    ):
    f = open(file_name, 'r')
    f.readline()
    label_list = list()
    time_list = list()
    for line in f:
        label, time = line[:-1].split(', ')
        label_list.append(label)
        time_list.append(time)
    
    plot_bar(
        x_list = label_list,
        y_list = time_list,
        x_label="Method",
        y_label = "Training Time per Epoch",
        figure_name =f"{benchmark_name} Running Time",
        figure_save_name=f"{benchmark_name}_running_time",
    )


def plot_mc_trajectory(
        x_list_list,
        y_list_list,
        x_label,
        y_label,
        label_name_list,
        figure_name,
        figure_save_name=None,
    ):
    color_list = ['g', 'b']
    fig, ax = plt.subplots(figsize=(6, 5))

    for idx, y_list in enumerate(y_list_list):
        sns.scatterplot(ax=ax, x=x_list_list[idx], y=y_list, label=label_name_list[idx], color=color_list[idx])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(figure_name)
    # plt.grid()
    plt.savefig(f"all_figures/{figure_save_name}.png")
    plt.close()



def extract_mountain_car(
        file_name_list,
        method_name_list,
        benchmark_name,
    ):
        p_list_list = list()
        v_list_list = list()
        label_name_list = list()

        for idx, file_name in enumerate(file_name_list):
            f = open(file_name, 'r')
            f.readline()
            f.readline()
            f.readline()
            p_list = list()
            v_list = list()
            for line in f:
                data = line[:-1].split(';')
                p = float(data[1].split(', ')[0])
                v = float(data[2].split(', ')[0])
                p_list.append(p)
                v_list.append(v)
            p_list_list.append(p_list)
            v_list_list.append(v_list)
            label_name_list.append(method_name_list[idx])
        
        plot_mc_trajectory(
            x_list_list=p_list_list,
            y_list_list=v_list_list,
            x_label='Position',
            y_label='Velocity',
            label_name_list=label_name_list,
            figure_name=f"{benchmark_name} Trajectory",
            figure_save_name=f"{benchmark_name}_trajectory"
        )



if __name__ == "__main__":
    # plot_loss('loss/') # the q and c loss
    # plot_loss_2('loss/')
    # plot_sample('data/sample_time.txt')
    # lr_bs_epoch_samplesize
    # plot_training_loss('gpu_DSE/result/mountain_car_DSE_0.001_2_10_400_True_10_100_1000_all_linearrelu_5_True_100.0_[80.0, 85.0, 90.0, 95.0]_0.1.txt', benchmark='mountain_car_DSE_0.001_2_10_400_True_10_100_1000_all_linearrelu_5_True_100.0_[80.0, 85.0, 90.0, 95.0]_0.1', method_name='DSE', log=False)
    # plot_vary_constraint('result/vary_constraint_volume_vs_point_sampling.txt')
    # vary_safe_bound()
    # abstract_component()
    # extract_verify_result()
    # provable_safe(
    #     benchmark_name='mountain_car',
    #     mode='provable_safety',
    #     p_list=[0.1, 0.5],
    # )
    # empirical_safe(
    #     benchmark_name='mountain_car',
    #     mode='empirical_test',
    #     p_list=[0.0, 0.1, 0.5],
    # )
    # extract_trajectory(
    #     mode='DiffAI',
    #     file_name='mountain_car_[30]_DiffAI_0.01_2_10_400_False_10_128_1000_all_linearrelu_no_act_5_True_[[-0.8, 0.8], [0.5, 10000.0]]_0.03_0.11_0.02_[0.0, 0.1]_[1.0, 0]_True_False_True_False_40_True_1_5_1.0_1e-06_True_True_True_10_1_True_False__0.05_0',
    #     name_list=['acceleration', 'position', 'velocity'],
    #     name_idx=0,
    # )
    # extract_trajectory(
    #     mode='DSE',
    #     file_name='mountain_car_[30]_DSE_0.01_2_10_400_True_10_128_1000_all_linearrelu_no_act_5_True_[[-0.8, 0.8], [0.5, 10000.0]]_0.2_1.1_0.1_[0.1, 0.1]_[1.0, 0]_True_False_True_False_40_True_0_3_1.0_1e-06_True_500_1_True_True__0.4_0',
    #     name_list=['acceleration', 'position', 'velocity'],
    #     name_idx=0,
    # )

    # plot test results,  unsafe probability, data loss
    # extract_test_info(
    #     file_name_list = [
    #         'gpu_DiffAI/result/thermostat_[30]_DiffAI_0.001_2000_30_800_False_2000_128_1000_all_linearrelu_5_True_83.5_92.0_0.5_[0.0]_True_False_40_0_15_1.0_1e-06_normal_55.0_70.0_True_True_evaluation.txt',
    #         'gpu_DiffAI/result/thermostat_[30]_DiffAI_0.01_10_30_800_False_10_128_1000_all_linearrelu_5_True_83.5_92.0_0.5_[0.0]_True_False_40_0_15_1.0_1e-06_normal_55.0_70.0_True_True_evaluation.txt',
    #         'gpu_DSE/result/thermostat_[30]_DSE_0.01_1_30_800_True_1_128_1000_all_linearrelu_5_True_83.5_92.0_0.5_[0.0]_True_False_40_0_15_1.0_1e-06_normal_55.0_70.0_True_True_evaluation.txt',
    #         'gpu_DSE/result/thermostat_[30]_DSE_0.01_10_30_800_True_10_128_1000_all_linearrelu_5_True_83.5_92.0_0.5_[0.0]_True_False_40_0_15_1.0_1e-06_normal_55.0_70.0_True_True_evaluation.txt'
    #     ],
    #     method_name_list = [
    #         'DiffAI(2k)',
    #         'DiffAI(10)',
    #         'DSE(10)',
    #         'DSE(1)',
    #     ],
    #     benchmark_name='Thermostat',
    # )

    # extract_running_time(
    #     file_name = 'all_results/thermostat_running_time.txt',
    #     benchmark_name='Thermostat',
    # )

    extract_mountain_car(
        file_name_list = [
            'gpu_DSE/result_test/trajectory/mountain_car_[30]_DSE_0.01_10_160_200_True_10_128_1000_all_linearrelu_no_act_5_False_1.4_0.8_-0.1_[0.0, 0.1]_True_False_40_0_6_1.0_1e-06_normal_-1.6_-0.0_True_2000_1_True_True__1.4_0_point.txt',
            'gpu_DiffAI/result_test/trajectory/mountain_car_[30]_DiffAI_0.01_50_40_200_False_50_128_1000_all_linearrelu_no_act_5_False_1.4_0.8_-0.1_[0.0, 0.1]_True_False_40_0_6_1.0_1e-05_normal_-1.6_-0.0_True_2000_1_True_True__1.4_0_point.txt',
        ]
        method_name_list = [
            'DSE',
            'DiffAI',
        ]
        benchmark_name='Mountain-Car(1.4)'
    )



