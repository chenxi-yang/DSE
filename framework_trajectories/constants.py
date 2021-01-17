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
B = var(1000) # the range of lambda

CURRENT_PROGRAM = 'program6_loop' # 'program_test_disjunction_2'

# PROGRAM #1
# ! have problem!
# x_l = [65.0]
# x_r = [75.0]
# target_theta = 61.7
# theta_l = 61.0
# theta_r = 63.5
# safe_l = 61.28 # 63.5 (original) # 64.534 (very tight)  
# safe_r = 82.80 # 83.59 (original) # 82.994 (very tight)  

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

# PROGRAM #4 [work]
# sample size: 500
# command: --lr 0.1 --stop_val 1.0 --optimizer gd_direct_noise
# x_l = [8.0] # initial height
# x_r = [12.0]
# target_theta = 5.0
# theta_l = 1.0
# theta_r = 6.0
# safe_l = 3.0
# safe_r = 9.3

# PROGRAM #5
# ! have problem
# x_l = [62.0]
# x_r = [72.0]
# target_theta = 57.046 # 57.7
# theta_l = 55.0
# theta_r = 58.3
# safe_l = 57.65 # 57.69 # 69.8
# safe_r = 76.12 # 76.76 # 77.0


# PROGRAM_TEST_DISJUNCTION
# x_l = [2.0]
# x_r = [9.99]
# target_theta = 5.49
# theta_l = 2.0
# theta_r = 9.0
# safe_l = 0.0
# safe_r = 11.0


# PROGRAM_TEST_DISJUNCTION_2
# x_l = [2.0]
# x_r = [9.99]
# target_theta = 5.49
# theta_l = 4.0
# theta_r = 9.0
# safe_l = N_INFINITY.data.item()# 0.0
# safe_r = 26.48

# PROGRAM_6
# x_l = [0.0, 0.0, 0.0, 0.0]
# x_r = [2.0, 2.0, 2.0, 2.0]
# target_theta = 30.0181 # 0.0055
# theta_l = 25.0 #0 .001
# theta_r = 35.0 # 0.01
# safe_l = 0.0 # N_INFINITY.data.item()
# safe_r = 2.01

# PROGRAM_6_loop
# sample size: 10000
# noise: 0.3
# command: --lr 0.1 --stop_val 0.01 --optimizer gd_direct_noise
# x_l = [0.0, 0.0, 0.0, 0.0]
# x_r = [1.0, 2.0, 2.0, 2.0]
# target_theta = 3.8
# theta_l = 2.0
# theta_r = 5.0
# safe_l = -0.008 # N_INFINITY.data.item()
# safe_r = 0.99342

#PROGRAM_7
x_l = [-5, -5, 0.0, 2.0, 2.5, 0.0]
x_r = [5, 5, 0.0, 2.0, 2.5, 2.0]
target_theta = 2.175
theta_l = 1.5
theta_r = 3.5
safe_l = -6.92 #-0.4
safe_r = 5.0 #0.5

# #PROGRAM_8
# x_l = [-0.2, 0.0, 0.0, 2.0, 2.5, 0.0]
# x_r = [0.1, 0.0, 0.0, 2.0, 2.5, 2.0]
# target_theta = 4.2
# theta_l = 2.0
# theta_r = 5.0
# safe_l = -2.61 #-0.4
# safe_r = 1.905 #0.5



# args
dataset_size = 50
lambda_ = 100.0

# w = 0.8

eta = 10.0
gamma = 0.55
alpha_coeff = 0.9

noise = 0.1 # 0.1