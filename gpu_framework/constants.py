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
real_unsafe_value =  args.real_unsafe_value
only_data_loss = args.only_data_loss
data_bs = args.data_bs
fixed_dataset = args.fixed_dataset
cuda_debug = args.cuda_debug

sound_verify = args.sound_verify
unsound_verify = args.unsound_verify
assert(test_mode == (sound_verify or unsound_verify))

# thermostat: 0.3
# mountain_car: 0.01


STATUS = 'Training' # a global status, if Training: use normal module, if Verifying: use sound module

path_num_list = [50]

K_DISJUNCTS = 10000000
SAMPLE_SIZE = 500
DOMAIN = "interval" # [interval, zonotope]

CURRENT_PROGRAM = 'program' + benchmark_name # 'program_test_disjunction_2'
DATASET_PATH = f"dataset/{benchmark_name}_{data_attr}.txt"
MODEL_PATH = f"gpu_{mode}/models"

# Linear nn, Sigmoid
if benchmark_name == "thermostat":
    x_l = [55.0]
    x_r = [62.0]
    # SAFE_RANGE = [55.0, 81.34] # strict
    SAFE_RANGE = [53.0, 82.8]
    # first expr
    # safe_range_upper_bound_list = np.arange(82.0, 83.0, 0.1).tolist()
    # PHI = 0.05 # unsafe probability
    # safe_range_upper_bound_list = np.arange(82.5, 83.0, 0.15).tolist()
    safe_range_upper_bound_list = np.arange(82.81, 82.999, 0.046).tolist()

    PHI = 0.10
    # SAFE_RANGE = [53.0, 82.0]
    # SAFE_RANGE = [52.0, 83.0] # not that strict
    # SAFE_RANGE = [50.0, 85.0] # not that loose

if benchmark_name == "mountain_car":
    x_l = [-0.6]
    x_r = [-0.4]

    # u,  p
    safe_range_list = [[-0.8, 0.8], [0.5, 10000.0]]
    phi_list = [0.1, 0.1]
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
    safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()

    # SAFE_RANGE = [100.0, 100.0]
    # safe_range_upper_bound_list = np.arange(80.0, 96.0, 5.0).tolist()
    # PHI = 0.1

# if adaptive_weight:
#     model_name_prefix = f"{benchmark_name}_{data_attr}_{n}_{lr}_{use_smooth_kernel}_{w_list}"
# else:
model_name_prefix = f"{benchmark_name}_{data_attr}_{n}_{lr}_{nn_mode}_{module}_{use_smooth_kernel}_{w_list}"
model_name_prefix = f"{model_name_prefix}_{outside_trajectory_loss}_{only_data_loss}_{data_bs}"
if fixed_dataset:
    model_name_prefix = f"{model_name_prefix}_{fixed_dataset}"

dataset_path_prefix = f"dataset/{benchmark_name}_{dataset_distribution}_{x_l[0]}_{x_r[0]}"

# args
dataset_size = 50
lambda_ = 100.0

# for debugging
TEST = False

PROTECTION_LOOP_NUM = 999
PROTECTION_LOOP_NUM_SMOOTH = 999

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

if not debug and not generate_all_dataset:
    result_prefix = f"{benchmark_name}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{safe_range_list}_{safe_range_bound_list}_{phi_list}_{w_list}_{outside_trajectory_loss}_{only_data_loss}_{sound_verify}_{unsound_verify}_{data_bs}"
    if fixed_dataset:
        result_prefix = f"{result_prefix}_{fixed_dataset}"
    if test_mode:
        # if outside_trajectory_loss:
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
    
    log_file = open(file_dir, 'w')
    log_file.write(f"{args}\n")
    log_file.write(f"Target info: {safe_range_list}, {phi_list}, \
        {w_list}, {method_list}, {safe_range_bound_list}\n")
    log_file.write(f"path_num_list: {path_num_list}")
    log_file.close()

    log_file_evaluation =  open(file_dir_evaluation, 'w')
    log_file_evaluation.write(f"{args}\n")
    log_file_evaluation.write(f"Target info: {safe_range_list}, {phi_list}, \
        {w_list}, {method_list}, {safe_range_bound_list}\n")
    log_file_evaluation.write(f"path_num_list: {path_num_list}")
    log_file_evaluation.close()
