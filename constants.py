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
early_stop = args.early_stop
quick_mode = args.quick_mode

data_attr = args.data_attr
# thermostat: normal_55.0_62.0
# mountain_car: normal_-0.6_-0.4
# print(f"test_mode: {test_mode}")

mode = args.mode
debug = args.debug
debug_verifier = args.debug_verifier
run_time_debug = args.run_time_debug
plot = args.plot
profile = args.profile

data_bs = args.data_bs

use_hoang = args.use_hoang
bound_start = args.bound_start
bound_end = args.bound_end

simple_debug = args.simple_debug
score_f = args.score_f
if score_f == 'hybrid':
    importance_weight_list = [1.0, 1.0] # 0.9 for volume, 0.1 for probability

extract_one_trajectory = args.extract_one_trajectory

AI_verifier_num_components = args.AI_verifier_num_components
SE_verifier_num_components = args.SE_verifier_num_components
# TODO: SE_verifier_num_components
SE_verifier_run_times = args.SE_verifier_run_times
train_sample_size = args.train_sample_size

K_DISJUNCTS = 10000000
SAMPLE_SIZE = train_sample_size
DOMAIN = "interval" # [interval, zonotope]

MODEL_PATH = f"gpu_{mode}/models"
map_mode = False
multi_agent_mode = False

status = ''

###################################################################################
####### Below are configurations for Thermostat, AC, Racetrack and Cartpole #######
###################################################################################
'''
new benchmark configuration is in the form of:
if benchmark_name == "new_benchmark_name":
    x_l = [x1_l, x2_l, ...] # the left bound for each input variable
    x_r = [x1_r, x2_r, ...] # the right bound for each input variable
    safe_range_list = [[o1_l, o1_r], ...] # the output range bound for each output variable
    # oi_l and oi_r is the left and right bound for the i-th output variable.
    w_list = [1.0] # default setting, this is not usually used when creating new benchmarks
    method_list = ['all'] # default setting, this is not usually used when creating new benchmarks
    name_list = ['x'] # default setting, for the variable name

    # Set the sequence of safe constraint (for right bound)
    safe_range_start=xx
    safe_range_end=xx
    safe_range_step=xx
    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]
'''

if benchmark_name == "thermostat_new":
    x_l = [60.0]
    x_r = [64.0]
    safe_range_list = [[55.0, 83.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=83.0
    safe_range_end=83.5
    safe_range_step=0.5
    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "aircraft_collision_new":
    x_l = [12.0]
    x_r = [16.0]
    # safe_range_list = [[40.0, 100000.0]]
    map_mode = True
    map_safe_range = [[[0.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']
    safe_range_list = [0]

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = [100000.0]


if benchmark_name == "racetrack_relaxed_multi":
    # two agents start from one point, 
    # they should be no-crash and the distance between two agents should be larger than 0.5 except the first one
    x_l = [5.0]
    x_r = [6.0]
    safe_range_list = [0]
    map_mode = True
    multi_agent_mode = True
    # y's range
    map_safe_range = [
        [[5.0, 6.0]], # the first step
        [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]],
        [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]],
        [[4.0, 9.0]], [[4.0, 9.0]], [[4.0, 9.0]], [[4.0, 9.0]],
        [[4.0, 10.0]],[[4.0, 10.0]],[[4.0, 10.0]],[[0.0, 10.0]], 
        [[0.0, 10.0]],[[0.0, 10.0]],[[0.0, 10.0]],[[0.0, 4.0]],
    ]
    distance_safe_range = [
        [[0.0, 10000.0]],
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]],
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], 
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]],
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]],
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    method_list = ['map_each', 'map_each', 'map_each'] # each element in the trajectory is 
    name_list = ['distance', 'position1', 'position2']
    safe_range_bound_list = [0]


if benchmark_name == "cartpole_v2":
    x_l = [-0.05, -0.05, -0.05, -0.05]
    x_r = [0.05, 0.05, 0.05, 0.05]
    safe_range_list = [[-0.1, 0.1]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=0.1
    safe_range_end=0.2
    safe_range_step=0.1

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


###################################################################
####### Below are other configurations for other benchmarks #######
###################################################################


if benchmark_name == "thermostat":
    x_l = [55.0] # the left bound
    x_r = [70.0] # the right bound
    safe_range_list = [[50.0, 82.0]]
    w_list = [1.0] # default setting
    method_list = ['all'] # default setting
    name_list = ['x'] # default setting

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


if benchmark_name == "pattern_example":
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

    safe_range_list = [[-10000.0, 1.0]]
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
        [[4.0, 6.0]],
        [[4.0, 6.0]], [[4.0, 6.0]], [[4.0, 6.0]], [[4.0, 6.0]],
        [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]],
        [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]],
        [[4.0, 9.0]], [[4.0, 9.0]], [[4.0, 9.0]], [[0.0, 9.0]], 
        [[0.0, 9.0]], [[0.0, 9.0]], [[0.0, 9.0]], [[0.0, 3.0]],
    ]
    # another constraint: distance > 0.5
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_easy_classifier_ITE":
    x_l = [4.0]
    x_r = [6.0]
    safe_range_list = [0]
    map_mode = True
    # y's range
    map_safe_range = [
        [[4.0, 6.0]], [[4.0, 6.0]], [[4.0, 6.0]], [[4.0, 6.0]],
        [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]],
        [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]],
        [[4.0, 9.0]], [[4.0, 9.0]], [[4.0, 9.0]], [[0.0, 9.0]], 
        [[0.0, 9.0]], [[0.0, 9.0]], [[0.0, 9.0]], [[0.0, 3.0]],
    ]
    distance_safe_range = []
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_easy_multi":
    # two agents start from one point, 
    # they should be no-crash and the distance between two agents should be larger than 0.5 except the first one
    x_l = [4.0]
    x_r = [6.0]
    safe_range_list = [0]
    map_mode = True
    multi_agent_mode = True
    # y's range
    map_safe_range = [
        [[4.0, 6.0]], # the first step
        [[4.0, 6.0]], [[4.0, 6.0]], [[4.0, 6.0]], [[4.0, 6.0]],
        [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]],
        [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]],
        [[4.0, 9.0]], [[4.0, 9.0]], [[4.0, 9.0]], [[0.0, 9.0]], 
        [[0.0, 9.0]], [[0.0, 9.0]], [[0.0, 9.0]], [[0.0, 3.0]],
    ]
    distance_safe_range = [
        [[0.0, 10000.0]],
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]],
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], 
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]],
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]],
        [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]], [[0.5, 10000.0]],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    method_list = ['map_each', 'map_each', 'map_each'] # each element in the trajectory is 
    name_list = ['distance', 'position1', 'position2']
    safe_range_bound_list = [0]



if benchmark_name == "racetrack_relaxed_multi2":
    # two agents start from one point, 
    # they should be no-crash and the distance between two agents should be larger than 0.5 except the first one
    x_l = [5.0]
    x_r = [6.0]
    safe_range_list = [0]
    map_mode = True
    multi_agent_mode = True
    # y's range
    map_safe_range = [
        [[5.0, 6.0]], # the first step
        [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]],
        [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]],
        [[4.0, 9.0]], [[4.0, 9.0]], [[4.0, 9.0]], [[4.0, 9.0]],
        [[4.0, 10.0]],[[4.0, 10.0]],[[4.0, 10.0]],[[0.0, 10.0]], 
        [[0.0, 10.0]],[[0.0, 10.0]],[[0.0, 10.0]],[[0.0, 4.0]],
    ]
    distance_safe_range = [
        [[0.0, 10000.0]],
        [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]],
        [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]], 
        [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]],
        [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]],
        [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    method_list = ['map_each', 'map_each', 'map_each'] # each element in the trajectory is 
    name_list = ['distance', 'position1', 'position2']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_easy_multi2":
    # two agents start from one point, 
    # they should be no-crash and the distance between two agents should be larger than 0.5 except the first one
    x_l = [4.0]
    x_r = [6.0]
    safe_range_list = [0]
    map_mode = True
    multi_agent_mode = True
    # y's range
    map_safe_range = [
        [[4.0, 6.0]], # the first step
        [[4.0, 6.0]], [[4.0, 6.0]], [[4.0, 6.0]], [[4.0, 6.0]],
        [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]],
        [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]], [[4.0, 8.0]],
        [[4.0, 9.0]], [[4.0, 9.0]], [[4.0, 9.0]], [[0.0, 9.0]], 
        [[0.0, 9.0]], [[0.0, 9.0]], [[0.0, 9.0]], [[0.0, 3.0]],
    ]
    distance_safe_range = [
        [[0.0, 10000.0]],
        [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], 
        [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], 
        [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], 
        [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5]], 
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    method_list = ['map_each', 'map_each', 'map_each'] # each element in the trajectory is 
    name_list = ['distance', 'position1', 'position2']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_moderate_classifier_ITE":
    x_l = [7.0]
    x_r = [10.0]
    safe_range_list = [0]
    map_mode = True
    # y's range
    map_safe_range = [
        [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[8.0, 10.0]], [[8.0, 10.0]],
        [[7.0, 10.0]], [[3.0, 10.0]], [[3.0, 10.0]], [[2.0, 9.0]],  [[1.0, 7.0]],
        [[1.0, 5.0]],  [[1.0, 5.0]],  [[2.0, 7.0]],  [[3.0, 9.0]],  [[3.0, 10.0]],
        [[5.0, 10.0]], [[7.0, 10.0]], [[8.0, 10.0]], [[7.0, 9.0]],  [[5.0, 9.0]],
        [[3.0, 8.0]],  [[2.0, 7.0]],  [[2.0, 6.0]],  [[2.0, 5.0]],  [[2.0, 7.0]], 
        [[2.0, 7.0]],  [[2.0, 7.0]],  [[4.0, 9.0]],  [[5.0, 9.0]],  [[6.0, 9.0]],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_moderate_2_classifier_ITE":
    x_l = [7.0]
    x_r = [10.0]
    safe_range_list = [0]
    map_mode = True
    # y's range
    map_safe_range = [
        [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]],
        [[6.0, 10.0]], [[3.0, 10.0]], [[3.0, 10.0]], [[2.0, 9.0]],  [[1.0, 7.0]],
        [[1.0, 5.0]],  [[1.0, 5.0]],  [[2.0, 7.0]],  [[3.0, 9.0]],  [[3.0, 10.0]],
        [[5.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_moderate_3_classifier_ITE":
    x_l = [7.0]
    x_r = [9.0]
    safe_range_list = [0]
    map_mode = True
    # y's range
    map_safe_range = [
        [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]],
        [[6.0, 10.0]], [[3.0, 10.0]], [[3.0, 10.0]], [[2.0, 9.0]],  [[1.0, 7.0]],
        [[1.0, 5.0]],  [[1.0, 5.0]],  [[2.0, 7.0]],  [[3.0, 9.0]],  [[3.0, 10.0]],
        [[5.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]], [[7.0, 10.0]],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_hard_classifier_ITE":
    x_l = [4.0]
    x_r = [6.0]
    safe_range_list = [0]
    map_mode = True
    # y's range
    map_safe_range = [
        [[4.0, 6.0]], [[3.0, 7.0]], [[2.0, 4.0], [5.0, 8.0]], [[0.0, 4.0], [5.0, 10.0]], [[1.0, 4.0], [5.0, 9.0]],
        [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]],
        [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]],
        [[1.0, 3.0], [6.0, 9.0]], [[1.0, 4.0], [5.0, 8.0]], [[1.0, 7.0]], [[2.0, 7.0]], [[3.0, 6.0]],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_easy_classifier":
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


if benchmark_name == "racetrack_easy_1_classifier":
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
        [0.0, 9.0], [0.0, 9.0], [0.0, 9.0], [0.0, 5.0],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_easy_2_classifier":
    x_l = [4.0]
    x_r = [6.0]
    safe_range_list = [0]
    map_mode = True
    # y's range
    map_safe_range = [
        [3.0, 6.0], [3.0, 6.0], [3.0, 6.0], [3.0, 6.0],
        [3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0],
        [3.0, 8.0], [3.0, 8.0], [3.0, 8.0], [3.0, 8.0],
        [3.0, 9.0], [3.0, 9.0], [3.0, 9.0], [0.0, 9.0], 
        [0.0, 9.0], [0.0, 9.0], [0.0, 9.0], [0.0, 5.0],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "racetrack_easy_1":
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
        [0.0, 9.0], [0.0, 9.0], [0.0, 9.0], [0.0, 5.0],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]

# ???
if benchmark_name == "racetrack_easy_sample":
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
        [0.0, 9.0], [0.0, 9.0], [0.0, 9.0], [0.0, 5.0],
    ]
    # map k-column in map[k] interval
    # 0 is the basic version
    w_list = [1.0]
    method_list = ['map_each'] # each element in the trajectory is 
    name_list = ['position']
    safe_range_bound_list = [0]


if benchmark_name == "thermostat_refined":
    x_l = [60.0]
    x_r = [64.0]
    safe_range_list = [[55.0, 83.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=83.0
    safe_range_end=83.5
    safe_range_step=0.5

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "cartpole_v1":
    x_l = [-0.05, -0.05, -0.05, -0.05]
    x_r = [0.05, 0.05, 0.05, 0.05]
    safe_range_list = [[-0.418, 0.418]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=0.418
    safe_range_end=0.518
    safe_range_step=0.1

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "cartpole_v2":
    x_l = [-0.05, -0.05, -0.05, -0.05]
    x_r = [0.05, 0.05, 0.05, 0.05]
    safe_range_list = [[-0.1, 0.1]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=0.1
    safe_range_end=0.2
    safe_range_step=0.1

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "cartpole_v3":
    x_l = [-0.05, -0.05, -0.05, -0.05]
    x_r = [0.05, 0.05, 0.05, 0.05]
    safe_range_list = [[-0.210, 0.210]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=0.210
    safe_range_end=0.309
    safe_range_step=0.1

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]





if benchmark_name == "aircraft_collision_refined_classifier":
    x_l = [12.0]
    x_r = [16.0]
    safe_range_list = [[40.0, 100000.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "thermostat_new_cnn":
    x_l = [60.0]
    x_r = [64.0]
    safe_range_list = [[55.0, 83.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=83.0
    safe_range_end=83.5
    safe_range_step=0.5

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "thermostat_new_tinyinput":
    x_l = [60.0]
    x_r = [60.1]
    safe_range_list = [[55.0, 83.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=83.0
    safe_range_end=83.5
    safe_range_step=0.5

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "thermostat_new_3branches":
    x_l = [60.0]
    x_r = [64.0]
    safe_range_list = [[55.0, 83.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=83.0
    safe_range_end=83.5
    safe_range_step=0.5

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "thermostat_new_40":
    x_l = [60.0]
    x_r = [64.0]
    safe_range_list = [[55.0, 83.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=83.0
    safe_range_end=83.5
    safe_range_step=0.5

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "thermostat_new_unsafe25":
    x_l = [60.0]
    x_r = [64.0]
    safe_range_list = [[55.0, 83.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=83.0
    safe_range_end=83.5
    safe_range_step=0.5

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "thermostat_new_unsafe50":
    x_l = [60.0]
    x_r = [64.0]
    safe_range_list = [[55.0, 83.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x']

    safe_range_start=83.0
    safe_range_end=83.5
    safe_range_step=0.5

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "aircraft_collision":
    x_l = [12.0]
    x_r = [16.0]
    safe_range_list = [[20.0, 100000.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "aircraft_collision_refined":
    x_l = [12.0]
    x_r = [16.0]
    safe_range_list = [[40.0, 100000.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "aircraft_collision_refined_classifier":
    x_l = [12.0]
    x_r = [16.0]
    safe_range_list = [[40.0, 100000.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "aircraft_collision_refined_classifier_ITE":
    x_l = [12.0]
    x_r = [16.0]
    safe_range_list = [[40.0, 100000.0]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = np.arange(safe_range_start, safe_range_end, safe_range_step).tolist()
    safe_range_bound_list = safe_range_bound_list[bound_start:bound_end]


if benchmark_name == "aircraft_collision_new":
    x_l = [12.0]
    x_r = [16.0]
    # safe_range_list = [[40.0, 100000.0]]
    map_mode = True
    map_safe_range = [[[0.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']
    safe_range_list = [0]

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = [100000.0]


if benchmark_name == "aircraft_collision_new_1":
    x_l = [12.0]
    x_r = [16.0]
    # safe_range_list = [[40.0, 100000.0]]
    map_mode = True
    map_safe_range = [[[0.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']
    safe_range_list = [0]

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = [100000.0]


if benchmark_name == "aircraft_collision_new_1_cnn":
    x_l = [12.0]
    x_r = [16.0]
    # safe_range_list = [[40.0, 100000.0]]
    map_mode = True
    map_safe_range = [[[0.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']
    safe_range_list = [0]

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = [100000.0]


if benchmark_name == "aircraft_collision_new_1_unsafe25":
    x_l = [12.0]
    x_r = [16.0]
    # safe_range_list = [[40.0, 100000.0]]
    map_mode = True
    map_safe_range = [[[0.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]],
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], 
                            [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]], [[40.0, 100000.0]]]
    w_list = [1.0]
    method_list = ['all']
    name_list = ['x1']
    safe_range_list = [0]

    safe_range_start=100000.0
    safe_range_end=100050.0
    safe_range_step=100

    safe_range_bound_list = [100000.0]

########################################################################

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
EPSILON = var(1e-6)
SMALL_PROBABILITY = var(0.01)
B = var(b) # the range of lambda

# w = 0.8

eta = 10.0
gamma = 0.55
alpha_coeff = 0.9

alpha_smooth_max = 0.8
eps = 1e-10

if benchmark_name is not None:
    model_name_prefix = f"{benchmark_name}_{nn_mode}_{l}_{data_bs}_{num_components}_{train_size}"
    if score_f != 'volume':
        model_name_prefix += f"_{score_f}"

    dataset_path_prefix = f"dataset/{benchmark_name}"

    expr_info_prefix = f"{train_size}_{safe_range_bound_list}"
    # TODO fix the name issue
    if score_f != 'hybrid':
        expr_info_prefix += f"_{score_f}"
    # test_info_prefix = f"{AI_verifier_num_components}_{SE_verifier_run_times}"
    test_info_prefix = f"{AI_verifier_num_components}"

    result_prefix = f"{model_name_prefix}_{expr_info_prefix}_{test_info_prefix}"

    if test_mode:
        file_dir = f"gpu_{mode}/result_test/{result_prefix}.txt"
        file_dir_evaluation = f"gpu_{mode}/result_test/{result_prefix}_evaluation.txt"
    else:
        file_dir = f"gpu_{mode}/result/{result_prefix}.txt"
        file_dir_evaluation = f"gpu_{mode}/result/{result_prefix}_evaluation.txt"

    trajectory_log_prefix = f"gpu_{mode}/result_test/trajectory/{result_prefix}_"

if not profile and not debug and not generate_dataset and not plot and not debug_verifier:
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
