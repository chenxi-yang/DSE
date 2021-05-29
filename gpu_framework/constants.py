from helper import *
from args import *

import numpy as np
# import domain

args = get_args()
generate_all_dataset = args.generate_all_dataset
dataset_distribution = args.dataset_distribution
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
# print(f"nn_mode: {nn_mode}")
b = args.b
module = args.module
num_components = args.num_components
verification_num_components = args.verification_num_components
verification_num_abstract_states = args.verification_num_abstract_states
use_smooth_kernel = args.use_smooth_kernel
save = args.save
test_mode = args.test_mode
adaptive_weight = args.adaptive_weight
outside_trajectory_loss = args.outside_trajectory_loss
verify_outside_trajectory_loss = args.verify_outside_trajectory_loss
# safe_start_idx = args.safe_start_idx
# safe_end_idx = args.safe_end_idx
# path_sample_size = args.path_sample_size
data_attr = args.data_attr
# thermostat: normal_55.0_62.0
# mountain_car: normal_-0.6_-0.4
# print(f"test_mode: {test_mode}")
mode = args.mode
debug = args.debug
perturbation_width = args.perturbation_width
real_unsafe_value = args.real_unsafe_value
only_data_loss = args.only_data_loss
data_bs = args.data_bs
fixed_dataset = args.fixed_dataset
cuda_debug = args.cuda_debug
use_data_loss = args.use_data_loss
data_safe_consistent = args.data_safe_consistent
use_hoang = args.use_hoang
bound_start = args.bound_start
bound_end = args.bound_end
sample_std = args.sample_std
sample_width = args.sample_width
# analyze_trajectory = args.analyze_trajectory
analysis = args.analysis
use_abstract_components = args.use_abstract_components
test_with_training = args.test_with_training
optimizer_method = args.optimizer_method
simple_debug = args.simple_debug

expr_i_number = args.expr_i_number

# print(f"sample_width: {sample_width}")
verify_use_probability = args.verify_use_probability
ini_unsafe_probability = args.ini_unsafe_probability

sound_verify = args.sound_verify
unsound_verify = args.unsound_verify
assert((test_mode or test_with_training) == (sound_verify or unsound_verify))

# thermostat: 0.3
# mountain_car: 0.01

STATUS = 'Training' # a global status, if Training: use normal module, if Verifying: use sound module

path_num_list = [30]

K_DISJUNCTS = 10000000
SAMPLE_SIZE = 500
DOMAIN = "interval" # [interval, zonotope]

CURRENT_PROGRAM = 'program' + benchmark_name # 'program_test_disjunction_2'
DATASET_PATH = f"dataset/{benchmark_name}_{data_attr}.txt"
MODEL_PATH = f"gpu_{mode}/models"

# Linear nn, Sigmoid
if benchmark_name == "thermostat":
    # x_l = [55.0]
    # x_r = [62.0]
    x_l = [55.0]
    x_r = [70.0]
    # SAFE_RANGE = [55.0, 81.34] # strict
    safe_range_list = [[50.0, 82.0]]
    phi_list = [0.0]
    phi_list[0] = ini_unsafe_probability
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']
    # SAFE_RANGE = [53.0, 82.8]
    # first expr
    component_bound_idx = 0
    bound_direction_idx = 1

    safe_range_start=83.5
    safe_range_end=92.0
    safe_range_step=0.5
    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    analysis_name_list = ['x']

if benchmark_name == "mountain_car":
    # x_l = [-0.6]
    # x_r = [-0.4]
    # x_l = [-1.2]
    # x_r = [-0.4]
    # x_l = [-0.6]
    # x_r = [0.0]
    # x_l = [-1.6]
    # x_r = [-0.0]
    # x_l = [-3.0]
    # x_r = [0.0]

    x_l =[-1.6]
    x_r = [-0.0]

    # u, p
    safe_range_list = [[-0.8, 0.8], [0.5, 10000.0]]
    phi_list = [0.0, 0.1]
    phi_list[0] = ini_unsafe_probability
    if adaptive_weight:
        w_list = [0.01, 0.99]
    else:
        # w_list = [0.4, 0.6]
        w_list = [1.0, 0]
    method_list = ['all', 'last']
    name_list = ['acceleration', 'position']
    # TODO: upper bound list:
    component_bound_idx = 0
    bound_direction_idx = 1 # left or right
    # safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()
    # safe_range_start=0.2
    # safe_range_end=1.1
    # safe_range_step=0.1
    safe_range_start=0.5
    safe_range_end=1.0 # 1.1
    safe_range_step=0.1
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['acceleration', 'position']

    # SAFE_RANGE = [100.0, 100.0]
    # safe_range_upper_bound_list = np.arange(80.0, 96.0, 5.0).tolist()
    # PHI = 0.1

if benchmark_name == "unsound_1":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 1.0]]
    phi_list = [0.0, 0.1]
    phi_list[0] = ini_unsafe_probability
    if adaptive_weight:
        w_list = [0.01]
    else:
        # w_list = [0.4, 0.6]
        w_list = [1.0]
    method_list = ['all']
    name_list = ['test']
    # TODO: upper bound list:
    component_bound_idx = 0
    bound_direction_idx = 1 # left or right
    # safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()
    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test']


if benchmark_name == "unsound_2_separate":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 2.5]]
    phi_list = [0.0, 0.1]
    phi_list[0] = ini_unsafe_probability
    if adaptive_weight:
        w_list = [0.01]
    else:
        # w_list = [0.4, 0.6]
        w_list = [1.0]
    method_list = ['all']
    name_list = ['test']
    # TODO: upper bound list:
    component_bound_idx = 0
    bound_direction_idx = 1 # left or right
    # safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()
    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test']


if benchmark_name == "unsound_2_overall":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 2.5]]
    phi_list = [0.0, 0.1]
    phi_list[0] = ini_unsafe_probability
    if adaptive_weight:
        w_list = [0.01]
    else:
        # w_list = [0.4, 0.6]
        w_list = [1.0]
    method_list = ['all']
    name_list = ['test']
    # TODO: upper bound list:
    component_bound_idx = 0
    bound_direction_idx = 1 # left or right
    # safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()
    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test']


if benchmark_name == "sampling_1":
    x_l = [-1.0]
    x_r = [1.0]

    safe_range_list = [[1.0, 2.5]]
    phi_list = [0.0, 0.1]
    phi_list[0] = ini_unsafe_probability
    if adaptive_weight:
        w_list = [0.01]
    else:
        # w_list = [0.4, 0.6]
        w_list = [1.0]
    method_list = ['all']
    name_list = ['test']
    # TODO: upper bound list:
    component_bound_idx = 0
    bound_direction_idx = 1 # left or right
    # safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()
    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test']


if benchmark_name == "sampling_2":
    x_l = [0.0]
    x_r = [10.0]

    safe_range_list = [[0.0, 0.0]]
    phi_list = [0.0, 0.1]
    phi_list[0] = ini_unsafe_probability
    if adaptive_weight:
        w_list = [0.01]
    else:
        # w_list = [0.4, 0.6]
        w_list = [1.0]
    method_list = ['all']
    name_list = ['test']
    # TODO: upper bound list:
    component_bound_idx = 0
    bound_direction_idx = 1 # left or right
    # safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()
    safe_range_start=0.0
    safe_range_end=0.1
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test']


if benchmark_name == "path_explosion":
    x_l = [2.0]
    x_r = [9.9]

    safe_range_list = [[4.0, 26.48]]
    phi_list = [0.0, 0.1]
    phi_list[0] = ini_unsafe_probability
    if adaptive_weight:
        w_list = [0.01]
    else:
        # w_list = [0.4, 0.6]
        w_list = [1.0]
    method_list = ['all']
    name_list = ['test']
    # TODO: upper bound list:
    component_bound_idx = 0
    bound_direction_idx = 1 # left or right
    # safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()
    safe_range_start=26.48
    safe_range_end=26.58
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test']


# if adaptive_weight:
#     model_name_prefix = f"{benchmark_name}_{data_attr}_{n}_{lr}_{use_smooth_kernel}_{w_list}"
# else:
data_attr = f"{dataset_distribution}_{x_l[0]}_{x_r[0]}"

model_name_prefix = f"{benchmark_name}_{data_attr}_{path_num_list}_{phi_list}_{n}_{lr}_{nn_mode}_{module}_{use_smooth_kernel}"
model_name_prefix = f"{model_name_prefix}_{data_bs}"
model_name_prefix = f"{model_name_prefix}_{bs}_{num_components}"
if benchmark_name in ["sampling_1", "unsound_1", "sampling_2"]:
    model_name_prefix = f"{model_name_prefix}_{l}"
if optimizer_method in ["SGD", "Adam-0"]:
    model_name_prefix += f"_{optimizer_method}"

if sample_std != 0.01:
    model_name_prefix += f"_{sample_std}"
if sample_width is not None:
    model_name_prefix += f"_{sample_width}"

if fixed_dataset:
    model_name_prefix = f"{model_name_prefix}_{fixed_dataset}"
# if not use_data_loss:
#     model_name_prefix = f"{model_name_prefix}_{use_data_loss}"

dataset_path_prefix = f"dataset/{benchmark_name}_{data_attr}"

# args
dataset_size = 50
lambda_ = 100.0

# for debugging
TEST = False

PROTECTION_LOOP_NUM = 999
PROTECTION_LOOP_NUM_SMOOTH = 999
# MAXIMUM_ITERATION = 50
if simple_debug:
    MAXIMUM_ITERATION = 5
else:
    MAXIMUM_ITERATION = 250

N_INFINITY = var(-10000.0)
P_INFINITY = var(10000.0)

INTERVAL_BETA = var(1.0) # 2.0
POINT_BETA = var(5.0) # var(50.0) # var(100.0) # 10.0s
PARTIAL_BETA = var(2.0) # 1.0
EPSILON = var(1e-10)
SMALL_PROBABILITY = var(0.01)
B = var(b) # the range of lambda

# w = 0.8

eta = 10.0
gamma = 0.55
alpha_coeff = 0.9

alpha_smooth_max = 0.8
eps = 1e-10

# result_prefix = f"{benchmark_name}_{path_num_list}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{safe_range_list}_{safe_range_bound_list}_{phi_list}_{w_list}_{outside_trajectory_loss}_{only_data_loss}_{sound_verify}_{unsound_verify}_{data_bs}"
result_prefix = f"{benchmark_name}_{path_num_list}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{safe_range_start}_{safe_range_end}_{safe_range_step}_{phi_list}_{sound_verify}_{unsound_verify}_{data_bs}"
result_prefix = f"{result_prefix}_{bound_start}_{bound_end}_{sample_std}_{sample_width}_{data_attr}"
# to avoid too long file name
if fixed_dataset:
    result_prefix = f"{result_prefix}_{fixed_dataset}"
# if not use_data_loss:
#     result_prefix = f"{result_prefix}_{use_data_loss}"
if test_with_training:
    result_prefix += f"_{test_with_training}"
if optimizer_method in ["SGD", "Adam-0"]:
    result_prefix += f"_{optimizer_method}"

if test_mode:
    # if outside_trajectory_loss:
    result_prefix = f"{result_prefix}_{verification_num_components}_{verification_num_abstract_states}_{verify_outside_trajectory_loss}_{verify_use_probability}"
    file_dir = f"gpu_{mode}/result_test/{result_prefix}.txt"
    file_dir_evaluation = f"gpu_{mode}/result_test/{result_prefix}_evaluation.txt"
    # else:
    #     file_dir = f"gpu_{mode}/result_test/{benchmark_name}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{safe_range_list}_{safe_range_bound_list}_{phi_list}_{w_list}.txt"
    #     file_dir_evaluation = f"gpu_{mode}/result_test/{benchmark_name}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{safe_range_list}_{safe_range_bound_list}_{phi_list}_{w_list}_evaluation.txt"
else:
    # if outside_trajectory_loss:
    file_dir = f"gpu_{mode}/result/{result_prefix}.txt"
    file_dir_evaluation = f"gpu_{mode}/result/{result_prefix}_evaluation.txt"
    # else:
    #     file_dir = f"gpu_{mode}/result/{benchmark_name}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{safe_range_list}_{safe_range_bound_list}_{phi_list}_{w_list}.txt"
    #     file_dir_evaluation = f"gpu_{mode}/result/{benchmark_name}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{safe_range_list}_{safe_range_bound_list}_{phi_list}_{w_list}_evaluation.txt"

trajectory_log_prefix = f"gpu_{mode}/result_test/trajectory/{result_prefix}_"

if not debug and not generate_all_dataset and not analysis:

    if os.path.exists(file_dir):
        log_file = open(file_dir, 'a')
    else:
        log_file = open(file_dir, 'w')    
    log_file.write(f"{args}\n")
    log_file.write(f"Target info: {safe_range_list}, {phi_list}, \
        {w_list}, {method_list}, {safe_range_bound_list}\n")
    log_file.write(f"path_num_list: {path_num_list}")
    log_file.close()

    if os.path.exists(file_dir):
        log_file_evaluation = open(file_dir_evaluation, 'a')
    else:
        log_file_evaluation =  open(file_dir_evaluation, 'w')
    log_file_evaluation.write(f"{args}\n")
    log_file_evaluation.write(f"Target info: {safe_range_list}, {phi_list}, \
        {w_list}, {method_list}, {safe_range_bound_list}\n")
    log_file_evaluation.write(f"path_num_list: {path_num_list}")
    log_file_evaluation.close()

