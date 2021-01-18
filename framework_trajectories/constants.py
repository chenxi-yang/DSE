from helper import *
# import domain

# 1, 2, 3, 4
mode_list = ['empty', 'interval', 'disjunction_of_intervals', 'partial_disjunction_of_intervals', 'disjunction_of_intervals_loop', 'disjunction_of_intervals_loop_sampling']
sample_list = ['direct_sampling', 'importance_sampling_scale', 'importance_sampling_translation', 'adaptive_importance_sampling']
MODE = 5
K_DISJUNCTS = 10000000
SAMPLE_SIZE = 500
SAMPLE_METHOD = 4
DOMAIN = "interval" # [interval, zonotope]
if MODE == 3: 
    MODE_NAME = mode_list[MODE] + '-' + 'Disjunct_' + str(K_DISJUNCTS) 
elif MODE == 4:
    MODE_NAME = mode_list[MODE] + '-' + 'SampleSize_' + str(SAMPLE_SIZE) 
elif MODE == 5:
    MODE_NAME = mode_list[MODE] + '_' + str(SAMPLE_SIZE) + '_' + str(sample_list[SAMPLE_METHOD - 1]) + '_' + DOMAIN
else: 
    MODE_NAME = mode_list[MODE]


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
POINT_BETA = var(100.0) # 10.0
PARTIAL_BETA = var(1.0) # 1.0
EPSILON = var(0.00001)
B = var(10000) # the range of lambda

CURRENT_PROGRAM = 'program7' # 'program_test_disjunction_2'

# PROGRAM #1
# ! have problem!
# x_l = [65.0]
# x_r = [75.0]
# target_theta = 61.7
# theta_l = 61.0
# theta_r = 63.5
# safe_l = 61.28 # 63.5 (original) # 64.534 (very tight)  
# safe_r = 82.80 # 83.59 (original) # 82.994 (very tight)  
'''
original setting: 
safe_l = 63.5
safe_r = 83.59
stop-val = 1.0
mcai: 10/10, avg loss: 0.59
baseline2: 10/10

update setting:
safe_l =  
safe_r = 
stop-val = 
mcai:     avg loss:
baseline2: 
'''


# PROGRAM #2 [work]
# sample size: 1000
# command: python run.py --lr 0.01 --stop_val 1.5 --optimizer gd_direct_noise
# x_l = [0.8, 1.6] # v1, v2
# x_r = [1.4, 2.0]
# target_theta = 5.6
# theta_l = 4.7
# theta_r = 5.8
# safe_l = 2.86 
# safe_r = 120 # P_INFINITY.data.item()
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
# command: --lr 0.1 --stop_val 1.0 --optimizer gd_direct_noise
# x_l = [9.0] # initial height
# x_r = [11.0]
# target_theta = 3.0
# theta_l = 1.0
# theta_r = 9.0
# safe_l = 2.368  
# safe_r = 7.04
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
mcai: 10/10 avg loss: 0.121
baseline2: 3/10
'''

# PROGRAM #4 [work]
# sample size: 500
# command: --lr 0.1 --stop_val 0.1 --optimizer gd_direct_noise
# x_l = [8.0] # initial height
# x_r = [12.0]
# target_theta = 5.0 
# theta_l = 1.0
# theta_r = 6.0
# safe_l = 3.0  
# safe_r = 9.3
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
# ! have problem
# x_l = [62.0]
# x_r = [72.0]
# target_theta = 57.046 # 57.7
# theta_l = 55.0
# theta_r = 58.3
# safe_l = 57.65 # 57.69 # 69.8
# safe_r = 76.12 # 76.76 # 77.0
'''
original setting: 
safe_l = 69.8
safe_r = 77.0
stop-val = 0.1
mcai: 10/10, avg loss: 0.032
baseline2: 9/10

update setting:
safe_l =  
safe_r = 
stop-val = 
mcai:     avg loss:
baseline2: 
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
# x_l = [2.0]
# x_r = [9.99]
# target_theta = 5.49
# theta_l = 5.0
# theta_r = 6.0 # 9.0
# safe_l = 4.0 # N_INFINITY.data.item()# 0.0
# safe_r = 26.48

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
# x_l = [0.0, 0.0, 0.0, 0.0]
# x_r = [1.0, 2.0, 2.0, 2.0]
# target_theta = 3.8
# theta_l = 3.0 #0 .001
# theta_r = 4.0 # 0.01
# safe_l = -0.062 # N_INFINITY.data.item()
# safe_r = 0.99342
'''
original setting: 
safe_l = N_INFINITY.data.item()
safe_r = 1.0
stop-val = 0.01
mcai: 10/10, avg loss: 0.0003
baseline2: 10/10

update setting:
sample size: 10000, 0.99
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
# x_l = [0.0, 0.0, 0.0, 0.0]
# x_r = [1.0, 2.0, 2.0, 2.0]
# target_theta = 3.8
# theta_l = 2.5
# theta_r = 5.0
# safe_l = -0.008 # N_INFINITY.data.item()
# safe_r = 0.99342
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
stop-val = 10. -> 1.5
mcai:     avg loss:
baseline2: 

refined initial partition[split 'y' into 10 equal partition]:
safe_l = -6.5992 
safe_r = 5.0
target_theta = 1.9
theta_l = 1.5
theta_r =  2.5 
mcai: avg loss:
baseline2: 2/10
'''


# #PROGRAM_8(Electronic Oscillator-Deep)
# stop-val 0.05
# x_l = [-5, -5, 0.0, 2.0, 2.5, 0.0]
# x_r = [5, 5, 0.0, 2.0, 2.5, 2.0]
# target_theta = 4.2
# theta_l = 1.0
# theta_r = 5.0
# safe_l = -1.895 # -2.61 #-0.4
# safe_r = 2.31 # 1.905 #0.5
'''
original setting: 
x_l = [-5, -5, 0.0, 2.0, 2.5, 0.0]
x_r = [5, 5, 0.0, 2.0, 2.5, 2.0]
target_theta = 4.2
theta_l = 1.0
theta_r = 5.0
safe_l = -1.895
safe_r = 2.31
stop-val = 1.0
mcai: 10/10, avg loss: 0.99
baseline2: 10/10

update setting:
safe_l = 
safe_r = 
stop-val = 
mcai:     avg loss:
baseline2: 
'''



# args
dataset_size = 50
lambda_ = 100.0

# w = 0.8

eta = 10.0
gamma = 0.55
alpha_coeff = 0.9

noise = 0.1 # 0.1