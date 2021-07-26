from args import *
from utils import *

import numpy as np
# import domain

args = get_args()
generate_dataset = args.generate_dataset
dataset_distribution = args.dataset_distribution
lr = args.lr
stop_val = args.stop_val
t_epoch = args.t_epoch
w = args.w
benchmark_name = args.benchmark_name
train_size = args.train_size
test_size = args.test_size
num_epoch = args.num_epoch
width = args.width
bs = args.bs
l = args.l
nn_mode = args.nn_mode
b = args.b
num_components = args.num_components
save = args.save
test_mode = args.test_mode

data_attr = args.data_attr
# thermostat: normal_55.0_62.0
# mountain_car: normal_-0.6_-0.4
# print(f"test_mode: {test_mode}")

mode = args.mode
debug = args.debug
debug_verifier = args.debug_verifier
run_time_debug = args.run_time_debug
plot = args.plot

data_bs = args.data_bs

use_hoang = args.use_hoang
bound_start = args.bound_start
bound_end = args.bound_end

simple_debug = args.simple_debug

extract_one_trajectory = args.extract_one_trajectory

AI_verifier_num_components = args.AI_verifier_num_components
SE_verifier_num_components = args.SE_verifier_num_components
# TODO: SE_verifier_num_components
SE_verifier_run_times = args.SE_verifier_run_times
train_sample_size = args.train_sample_size

# thermostat: 0.3
# mountain_car: 0.01

K_DISJUNCTS = 10000000
SAMPLE_SIZE = train_sample_size
DOMAIN = "interval" # [interval, zonotope]

MODEL_PATH = f"gpu_{mode}/models"
map_mode = False

status = ''

if benchmark_name == "thermostat":
    x_l = [55.0]
    x_r = [70.0]
    safe_range_list = [[50.0, 82.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=85.0
    safe_range_end=91.0
    safe_range_step=0.5

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]

if benchmark_name == "mountain_car":
    x_l = [-0.6]
    x_r = [-0.4]

    # safe_range[-1.0, x]
    # u
    safe_range_list = [[-1.0, 0.8]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['acceleration']

    safe_range_start=0.5
    safe_range_end=1.2 # 1.1
    safe_range_step=0.1
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]

if benchmark_name == "unsmooth_1":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]

if benchmark_name == "unsmooth_1_a":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]

if benchmark_name == "unsmooth_1_b":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[-10000.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]

if benchmark_name == "unsmooth_1_c":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[-10000.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern1_a":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern1_b":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[-10000, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern2":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[-10000, 0.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern3_a":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern3_b":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[-10000.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern31_a":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern31_b":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[-10000.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern5_a":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[1.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern5_b":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[-10000.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1.0
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern6":
    x_l = [-1.0]
    x_r = [1.0]

    safe_range_list = [[-5, 0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=0
    safe_range_end=0.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern7":
    x_l = [-1.0]
    x_r = [1.0]

    safe_range_list = [[-5, 0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=0
    safe_range_end=0.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "pattern8":
    x_l = [-5.0]
    x_r = [5.0]

    safe_range_list = [[-10000.0, 1.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['test']

    safe_range_start=1
    safe_range_end=1.5
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "racetrack_easy":
    x_l = [4.0]
    x_r = [6.0]
    safe_range_list = [0]
    map_mode = True
    # y's range
    map_safe_range = [
        [4.0, 6.0], [4.0, 6.0], [4.0, 6.0], [4.0, 6.0],
        [4.0, 7.0], [4.0, 7.0], [4.0, 7.0], [4.0, 7.0],
        [4.0, 8.0], [4.0, 8.0], [4.0, 8.0], [4.0, 8.0],
        [4.0, 9.0], [4.0, 9.0], [4.0, 9.0], [0.0, 9.0], 
        [0.0, 9.0], [0.0, 9.0], [0.0, 9.0], [0.0, 3.0],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


model_name_prefix = f"{benchmark_name}_{nn_mode}_{l}_{data_bs}_{num_components}"

dataset_path_prefix = f"dataset/{benchmark_name}"

# args
dataset_size = 50
lambda_ = 100.0

# for debugging
TEST = False

PROTECTION_LOOP_NUM = 999
PROTECTION_LOOP_NUM_SMOOTH = 999
MAXIMUM_ITERATION = 300
if plot and benchmark_name == "mountain_car":
    MAXIMUM_ITERATION = 600

N_INFINITY = var(-10000.0)
P_INFINITY = var(10000.0)

INTERVAL_BETA = var(0.5) # var(1.0) # 2.0
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

expr_info_prefix = f"{train_size}_{safe_range_bound_list}"
test_info_prefix = f"{AI_verifier_num_components}_{SE_verifier_run_times}"

result_prefix = f"{model_name_prefix}_{expr_info_prefix}_{test_info_prefix}"

if test_mode:
    file_dir = f"gpu_{mode}/result_test/{result_prefix}.txt"
    file_dir_evaluation = f"gpu_{mode}/result_test/{result_prefix}_evaluation.txt"
else:
    file_dir = f"gpu_{mode}/result/{result_prefix}.txt"
    file_dir_evaluation = f"gpu_{mode}/result/{result_prefix}_evaluation.txt"

trajectory_log_prefix = f"gpu_{mode}/result_test/trajectory/{result_prefix}_"


if not debug and not generate_dataset and not plot and not debug_verifier:
    if os.path.exists(file_dir):
        log_file = open(file_dir, 'a')
    else:
        log_file = open(file_dir, 'w')    
    log_file.write(f"{args}\n")
    log_file.write(f"Target info: {safe_range_list}, {safe_range_bound_list}\n")
    log_file.close()

    if os.path.exists(file_dir):
        log_file_evaluation = open(file_dir_evaluation, 'a')
    else:
        log_file_evaluation =  open(file_dir_evaluation, 'w')
    log_file_evaluation.write(f"{args}\n")
    log_file_evaluation.write(f"Target info: {safe_range_list}, {safe_range_bound_list}\n")
    log_file_evaluation.close()


# below are other benchmarks

if benchmark_name == "unsmooth_2_separate":
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


if benchmark_name == "unsmooth_2_overall":
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


if benchmark_name == "fairness_1":
    x_l = [0.0]
    x_r = [10.0]

    safe_range_list = [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
    phi_list = [0.0] * 4
    phi_list[0] = ini_unsafe_probability
    if adaptive_weight:
        w_list = [0.01]
    else:
        # w_list = [0.4, 0.6]
        w_list = [1.0] * 4
    method_list = ['all'] * 4
    name_list = ['test'] * 4
    component_bound_idx = 0
    bound_direction_idx = 1 # left or right
    # safe_range_bound_list = np.around(np.arange(0.5, 1.1, 0.1), 2).tolist()

    # seems no relationship with this
    safe_range_start=0.0
    safe_range_end=0.1
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test'] * 4


if benchmark_name == "path_explosion":
    x_l = [2.0]
    x_r = [4.8]

    safe_range_list = [[0.0, 5.0]]
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
    safe_range_start=5.0
    safe_range_end=5.2
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test']


if benchmark_name == "path_explosion_2":
    x_l = [2.0]
    x_r = [4.8]

    safe_range_list = [[1.0, 5.0]]
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
    safe_range_start=5.0
    safe_range_end=5.2
    safe_range_step=1.0
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['test']


if benchmark_name == "mountain_car_1":

    x_l =[-0.6]
    x_r = [-0.4]

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
    safe_range_start=0.07
    safe_range_end=0.08 # 1.1
    safe_range_step=0.1
    safe_range_bound_list = np.around(np.arange(safe_range_start, safe_range_end, safe_range_step), 2).tolist()
    analysis_name_list = ['acceleration', 'position']
