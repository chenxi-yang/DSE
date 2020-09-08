from helper import *

# 1, 2, 3
mode_list = ['interval', 'disjunction_of_intervals', 'partial_disjunction_of_intervals']
MODE = 1
K_DISJUNCTS = 5

N_INFINITY = var(-10000.0)
P_INFINITY = var(10000.0)

INTERVAL_BETA = var(2.0) # 2.0
POINT_BETA = var(10.0) # 10.0
PARTIAL_BETA = var(1.0) # 1.0
EPSILON = var(0.00001)

# PROGRAM #1
# x_l = [65.0]
# x_r = [75.0]
# target_theta = 70.0
# theta_l = 55.0
# theta_r = 80.0
# safe_l = 69.8
# safe_r = 77.0

# PROGRAM #2
# x_l = [2.9, 2.9]
# x_r = [3.1, 3.1]
# target_theta = 8.0
# theta_l = 6.0
# theta_r = 9.0
# safe_l = 3.0
# safe_r = P_INFINITY.data.item()

# PROGRAM #3
x_l = [9.0] # initial height
x_r = [11.0]
target_theta = 2.0
theta_l = 1.0
theta_r = 9.0
safe_l = 0.0
safe_r = 8.0

dataset_size = 50
lambda_ = 10.0

eta = 10.0
gamma = 0.55