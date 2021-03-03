from helper import *
from args import *

# import domain

args = get_args()
lr = args.lr
stop_val = args.stop_val
t_epoch = args.t_epoch
optimizer_name = args.optimizer
w = args.w
benchmark_id = args.benchmark_id
data_size = args.data_size
test_portion = args.test_portion
num_epoch = args.num_epoch
width = args.width
noise = args.noise
bs = args.bs
n = args.n
l = arg.l
nn_mode = args.nn_mode
# path_sample_size = args.path_sample_size

sample_size_list = [250, 100, 50, 10]
data_for_sample_list = [100]  # [100, 50, 20]

# 1, 2, 3, 4
mode_list = ['empty', 'interval', 'disjunction_of_intervals', 'partial_disjunction_of_intervals', 'disjunction_of_intervals_loop', 'disjunction_of_intervals_loop_sampling']
sample_list = ['direct_sampling', 'importance_sampling_scale', 'importance_sampling_translation', 'adaptive_importance_sampling']
MODE = 5
K_DISJUNCTS = 10000000
# SAMPLE_SIZE = path_sample_size # 500
# sample_size_list = [500, 250, 100, 50, 10]
SAMPLE_SIZE = 500
data_for_sample = 100
SAMPLE_METHOD = 3 # direct sampling  # 4: adaptive sampling
THETA_SAMPLE_SIZE = 10 # 1000
DOMAIN = "interval" # [interval, zonotope]
if MODE == 3: 
    MODE_NAME = mode_list[MODE] + '-' + 'Disjunct_' + str(K_DISJUNCTS) 
elif MODE == 4:
    MODE_NAME = mode_list[MODE] + '-' + 'SampleSize_' + str(SAMPLE_SIZE) 
elif MODE == 5:
    MODE_NAME = mode_list[MODE] + '_' + str(SAMPLE_SIZE) + '_' + str(sample_list[SAMPLE_METHOD - 1]) + '_' + DOMAIN
else: 
    MODE_NAME = mode_list[MODE]

# debug_file_dir = 'result/log'  + str(benchmark_id) + '_' + str(data_size) + '_' + str(test_portion) + '.txt'
# debug_file = open(log_file_dir, 'w')
# debug_file.close()

# for debugging
TEST = False

# for importance sampling translation : f(x) = f(x-c)
c = 1
# for importance sampling gradient: f(x) = f_i(x) - f_i-1(x), keep memory of k steps back
k = 1

PROTECTION_LOOP_NUM = 999
PROTECTION_LOOP_NUM_SMOOTH = 999

N_INFINITY = var(-10000.0)
P_INFINITY = var(10000.0)

INTERVAL_BETA = var(1.0) # 2.0
POINT_BETA = var(5.0) # var(50.0) # var(100.0) # 10.0s
PARTIAL_BETA = var(1.0) # 1.0
EPSILON = var(0.00001)
SMALL_PROBABILITY = var(0.01)
B = var(1000) # the range of lambda

CURRENT_PROGRAM = 'program' + str(benchmark_id) # 'program_test_disjunction_2'


# Linear nn, Sigmoid
if benchmark_id == 11:
# python run.py --lr 0.5 --stop_val 5.0 --optimizer gd_direct_noise --w 5.0 --data_size 20000 --test_portion 0.99 --benchmark_id 11
    x_l = [52.0]
    x_r = [59.0]
    target_theta = [66.0] + [0.0] * 9
    # Add one bias for hidden, or 
    theta_l = [59.1] + [0.8] + [-0.001] + [0.08] + [-0.001] * 3 + [-2.0] * 2 + [50.0]  # tOff, w11, w12, w21, w22, b1, b2, w3, w4
    theta_r = [67.0] + [1.0] + [0.001] + [0.11] + [0.001] * 3 + [5.0] * 2 + [60.0]
    #! the loosest constraint (both DSE and baseline1 can learn)
    # safe_l = 47.00 # 58.425
    # safe_r = 86.67 # 83.19
    #! constraint in between (testing)
    safe_l = 52.02 
    safe_r = 84.995
    #! the most strict constraint(both DSE and baseline1 can not learn)
    # safe_l = 53.00 # 47.00
    # safe_r = 83.00 # 83.67
    lr_l = [lr] + [lr/1e-3] * 6 + [lr/1e-2] * 2 + [lr]
    # noise = [10.0] + [100] * 8 + [10]


# Linear nn, no ReLu
if benchmark_id == 10:
# python run.py --lr 0.5 --stop_val 3.0 --optimizer gd_direct_noise --w 3.0 --data_size 20000 --test_portion 0.99 --benchmark_id 10
    x_l = [52.0]
    x_r = [59.0]
    target_theta = [62.0] + [0.0] * 8
    theta_l = [60.0] + [0.8] + [-0.001] + [0.08] + [-0.001] * 3 + [0.95] * 2 # tOff, w11, w12, w21, w22, b1, b2, w3, w4
    theta_r = [67.0] + [1.0] + [0.001] + [0.11] + [0.001] * 3 + [1.0] * 2
    safe_l = 51.9
    safe_r = 84.1
    lr_l = [lr] + [lr/1e-3] * 8
    noise = [0.1] + [100] * 8


# PROGRAM #1
# ! have problem!
if benchmark_id == 1:
    x_l = [52.0]
    x_r = [59.0]
    target_theta = 59.48
    # safe_l = 60.30 # (tight)
    # safe_r = 81.9513
    safe_l = 60.95 #(tighter, with 10 initial partition)
    safe_r = 81.606
    theta_l = 58.1
    theta_r = 65.0

'''

original setting: 
# sample size: 10000
target_theta = 59.48
safe_l = 60.0
safe_r = 83.0
stop-val = 0.1
mcai: 10/10, avg loss: 0.032
baseline2: 9/10

update setting: (tight constraint)
target theta: 59.48
safe_l = 60.30 # (tight)
safe_r = 81.9513
theta_l = 55.0
theta_r = 65.0
stop-val = 3.0
mcai:     avg loss:
baseline2: 5/10

refined initial partition[split 'x' into 10 equal partition]:
command: --lr 0.1 --stop_val 3.0 --optimizer gd_direct_noise --w 3.0
sample size: 10000
previous setting restrict the approximation but large point smooth loss
safe_l = 60.95
safe_r = 81.606
target_theta = 59.48s
theta_l = 55.0
theta_r = 62.0
stop-val = 3.0
mcai: 10/10 avg loss: 1.40710 (1.89937, 0.43740, 1.05413[not], 2.11605, 1.52856)
baseline2: 0/10
'''


# PROGRAM #2 [work]
# sample size: 1000
# command: python run.py --lr 0.01 --stop_val 1.5 --optimizer gd_direct_noise
if benchmark_id == 2:
    x_l = [0.8, 1.6] # v1, v2
    x_r = [1.4, 2.0]
    target_theta = 5.6
    theta_l = 4.7
    theta_r = 5.8
    safe_l = 2.86 
    safe_r = 120 # P_INFINITY.data.item()
'''
original setting: 
safe_l = 0.3, 
safe_r = P_INFINITY.data.item() 
mcai: 10/10 avg loss: 0.54
baseline2: 10/10

tight setting:
target_theta = 5.6
theta_l = 4.7
theta_r = 5.8
safe_l = 2.368  
safe_r = 7.04
mcai: 9/10 avg loss: 0.36
baseline2: 4/10
'''


# PROGRAM #3 [work]
# sample size: 1000
# command: --lr 0.01 --stop_val 1.0 --optimizer gd_direct_noise
if benchmark_id == 3:
    x_l = [9.0] # initial height
    x_r = [11.0]
    target_theta = 3.0
    theta_l = 1.0
    theta_r = 9.0
    safe_l = 2.368  
    safe_r = 7.04
'''
original setting: 
safe_l = 0.0
safe_r = 7.5
target_theta = 3.0
theta_l = 1.0
theta_r = 9.0
baseline2: 7/10

tight setting:
target_theta = 3.0
theta_l = 1.0
theta_r = 9.0
safe_l = 2.368  
safe_r = 7.04
mcai: 10/10  avg loss: 0.121
baseline2: 3/10
'''

# PROGRAM #4 [work]
# sample size: 500
# command: --lr 0.1 --stop_val 0.1 --optimizer gd_direct_noise
if benchmark_id == 4:
    x_l = [8.0] # initial height
    x_r = [12.0]
    target_theta = 5.0 
    theta_l = 1.0
    theta_r = 6.0
    safe_l = 3.0  
    safe_r = 9.3
'''
original setting: 
same
stop-val = 0.1

update setting:
stop-val = 1.0
safe_l = 3.0  
safe_r = 9.3
mcai: 10/10 avg loss: 0.0204
baseline2: 2/10
'''

# PROGRAM #5
if benchmark_id == 5:
    x_l = [62.0]
    x_r = [72.0]
    target_theta = 59.54 # 56.1 # 57.046 # 57.7
    theta_l = 55.0 # 55.9 # 55.0
    theta_r = 62.0 # 65.0 # 58.3
    safe_l = 61.04 # safe_l = 57.557# 57.69 # 69.8
    safe_r = 74.697 # safe_r = 74.8263 # 76.76 # 77.0

'''
original setting: 
safe_l = 69.8
safe_r = 77.0
stop-val = 0.1
mcai: 10/10, avg loss: 0.032
baseline2: 9/10

update setting:
target theta: 56.1
safe_l = 56.64
safe_r = 77.31
theta_l = 55.0
theta_r =  58.3
stop-val = 0.5
mcai:   10/10  avg loss:
baseline2: 8 /10

refined initial partition[split 'x' into 10 equal partition]:
command: --lr 0.1 --stop_val 1.0 --optimizer gd_direct_noise
previous setting restrict the approximation but large point smooth loss
safe_l = 61.04
safe_r = 74.697
target_theta = 59.54
theta_l = 55.0
theta_r = 62.0
stop-val = 1.0
mcai: 10/10 avg loss: 0.119622 (0.12925[not provably safe], 0.15411, 0.14261[not provably safe], 0.17214[not],  0.15115)
baseline2: 0/10
'''



# PROGRAM_TEST_DISJUNCTION 
# x_l = [2.0]
# x_r = [9.99]
# target_theta = 5.49
# theta_l = 2.0
# theta_r = 9.0
# safe_l = 0.0
# safe_r = 11.0

# PROGRAM_TEST_DISJUNCTION_2 [work]
# command: --lr 0.1 --stop_val 1.5 --optimizer gd_direct_noise
# sample size: plus one critical datapoint when checking
# large initial penalty: B=10000
if benchmark_id == 9:
    x_l = [2.0]
    x_r = [9.99]
    target_theta = 5.49
    theta_l = 5.0
    theta_r = 6.0 # 9.0
    safe_l = 4.0 # N_INFINITY.data.item()# 0.0
    safe_r = 26.48

'''
original setting: 
safe_l = 4.0 
safe_r = 26.48
stop-val = 1.0
mcai: 9/10, avg loss: 0.97
baseline2: 6/10

update setting:
sample size: plus one critical datapoint when checking
target_theta = 5.49
theta_l = 5.0
theta_r = 6.0 # 9.0
safe_l = 4.0
safe_r = 26.48
stop-val = 1.5
mcai: 9/10, avg loss: 1.02
baseline2: 2/10
'''


# PROGRAM_6
# command: python run_baseline_2.py --lr 0.1 --stop_val 0.01 --optimizer gd_direct_noise
if benchmark_id == 6:
    # x_l = [0.0, 0.0, 0.0, 0.0]
    # x_r = [1.0, 2.0, 2.0, 2.0]

    x_l = [0.0, 0.0, 0.0, 0.0]
    x_r = [1.0, 2.0, 2.0, 2.0]
    
    target_theta = 3.8
    theta_l = 3.0 #
    theta_r = 4.0 # 0.01
    safe_l = -0.062 # N_INFINITY.data.item()
    safe_r = 0.99342
'''
original setting: 
safe_l = N_INFINITY.data.item()
safe_r = 1.0
stop-val = 0.01
mcai: 10/10, avg loss: 0.0003
baseline2: 10/10

update setting:
sample size: 10000, 0.99
target_theta = 3.8
theta_l = 3.0 #0 .001
theta_r = 4.0 # 0.01
safe_l =  -0.062 # N_INFINITY.data.item()
safe_r = 0.99342
stop-val = 0.01
mcai:   10/10  avg loss: 0.003505 (0.00351, 0.00348, 0.00337, 0.00346, 0.00349， 0.00372)
baseline2: 5/10
'''


# PROGRAM_6_loop [work]
# sample size: 20000
# noise: 0.3
# command: --lr 0.1 --stop_val 0.01 --optimizer gd_direct_noise
if benchmark_id == 6.1:
    x_l = [0.0, 0.0, 0.0, 0.0]
    x_r = [1.0, 2.0, 2.0, 2.0]
    target_theta = 3.8
    theta_l = 2.5
    theta_r = 5.0
    safe_l = -0.008 # N_INFINITY.data.item()
    safe_r = 0.99342
'''
original setting: 
safe_l = N_INFINITY.data.item()
safe_r = 1.0
stop-val = 0.01
mcai: 10/10, avg loss: 0.00002
baseline2: 10/10s

update setting:
safe_l = -0.008 # N_INFINITY.data.item()
safe_r = 0.99342
stop-val = 0.01
mcai: 10/10, avg loss: 0.00004
baseline2: 3/10
'''


#PROGRAM_7(Electronic Oscillator-Loop)
# stop-val: 1.5 --lr 0.0000001
# sample size: 500
# x_l = [-5, -5, 0.0, 2.0, 2.5, 0.0]
# x_r = [5, 5, 0.0, 2.0, 2.5, 2.0]
# target_theta = 1.9 # , probability loss: 1.5
# theta_l = 1.5 # 1.5
# theta_r = 2.5
# safe_l = -8.0 #-0.4
# safe_r = 5.0 #0.5

# expr A: for log, upload to robovision
# x_l = [-5, -5, 0.0, 2.0, 2.5, 0.0]
# x_r = [5, 5, 0.0, 2.0, 2.5, 2.0]
# target_theta = 1.9 # 4.2, loss: < 1.0 # 2.175, probability loss: 1.5
# theta_l = 1.2 # 4.0 # 1.5
# theta_r =  4.5 # 4.5
# safe_l = -6.5992 #-0.4
# safe_r = 5.0 #0.5

# expr B: run 10 times, check safety
if benchmark_id == 7:
    x_l = [-5, -5, 0.0, 2.0, 2.5, 0.0]
    x_r = [5, 5, 0.0, 2.0, 2.5, 2.0]
    target_theta = 1.9 # 4.2, loss: < 1.0 # 2.175, probability loss: 1.5
    theta_l = 1.5 # 4.0 # 1.5
    theta_r =  2.5 # 4.5
    safe_l = -6.5992 #-0.4
    safe_r = 5.0 #0.5
# stop-val 1.0s

'''
original setting: 
safe_l = -8.0
safe_r = 5.0
stop-val = 1.0
# target_theta = 2.175, probability loss: 1.5
# theta_l = 1.5
# theta_r = 4.5
mcai: 10/10, avg loss: 0.022
baseline2: 10/10

update setting: [similar]
safe_l = -6.92
safe_r = 5.0  # can not be more refined
stop-val = 1. -> 1.5
mcai:     avg loss:
baseline2: 

refined initial partition[split 'y' into 10 equal partition]:
command: --lr 0.001 --stop_val 1.5 --optimizer gd_direct_noise
safe_l = -6.5992 
safe_r = 5.0
target_theta = 1.9
theta_l = 1.5
theta_r =  2.5 
mcai:  10/10 avg loss: 1.368698 (1.13999, 1.4434, 1.46932,  1.41152, 1.37926)
baseline2: 2/10
'''


# #PROGRAM_8(Electronic Oscillator-Deep)
# stop-val 0.05
if benchmark_id == 8:
    x_l = [-5, -5, 0.0, 2.0, 2.5, 0.0]
    x_r = [5, 5, 0.0, 2.0, 2.5, 2.0]
    target_theta = 1.634
    theta_l = 1.5
    theta_r = 2.5
    safe_l = -5.7658 # -2.61 #-0.4
    safe_r = 5.0 # 1.905 #0.5
'''
original setting: 
x_l = [-5, -5, 0.0, 2.0, 2.5, 0.0]
x_r = [5, 5, 0.0, 2.0, 2.5, 2.0]
target_theta = 4.2
theta_l = 1.0
theta_r = 5.0
safe_l = -1.895
safe_r = 2.31
stop-val = 0.5
mcai: 10/10, avg loss: 0.99
baseline2: 10/10

update setting:
sample size: 10000, 0.99
python run.py --lr 0.0001 --stop_val 1.0 --optimizer gd_direct_noise --w 1.0 --data_size 10000, --test_portion 0.99
target_theta = 1.634
safe_l = -5.7658
safe_r = 5.0
stop-val = 1.0
mcai: 0.1996    avg loss:(0.16272, 0.21603, 0.19318, 0.19341, 0.23274, 0.20775)
baseline2: 1/10

refined initial partition[split 'y' into 10 equal partition]:
safe_l = 
safe_r = 
target_theta = 
theta_l = 
theta_r =  
mcai: avg loss: 
baseline2: 
'''



# args
dataset_size = 50
lambda_ = 100.0

# w = 0.8

eta = 10.0
gamma = 0.55
alpha_coeff = 0.9

alpha_smooth_max = 0.8

file_dir =  f"result/thermostat_nn_volume_{safe_l}_{safe_r}_{lr}_{bs}_{num_epoch}.txt"
log_file = open(file_dir, 'w')
log_file.write(f"{args}\n")
log_file.write(f"sample_size_list: {sample_size_list}")
log_file.close()