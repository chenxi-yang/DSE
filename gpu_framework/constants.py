from helper import *
from args import *

import numpy as np
# import domain

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
# thermostat: normal_55.0_62.0
# mountain_car: normal_-0.6_-0.4
# print(f"test_mode: {test_mode}")
mode = args.mode
debug = args.debug
perturbation_width = args.perturbation_width
# thermostat: 0.3
# mountain_car: 0.01


model_name_prefix = f"{benchmark_name}_{data_attr}_{n}_{lr}_{use_smooth_kernel}"

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
    x_r =  [-0.4]
    SAFE_RANGE = [100.0, 100.0]
    safe_range_upper_bound_list = np.arange(80.0, 96.0, 5.0).tolist()
    PHI = 0.1
    


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

if not debug:
    if test_mode:
        file_dir = f"gpu_{mode}/result_test/{benchmark_name}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}.txt"
    else:
        file_dir = f"gpu_{mode}/result/{benchmark_name}_{mode}_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}.txt"
    log_file = open(file_dir, 'w')
    log_file.write(f"{args}\n")
    log_file.write(f"safe range: {SAFE_RANGE}\n")
    log_file.write(f"path_num_list: {path_num_list}")
    log_file.close()
