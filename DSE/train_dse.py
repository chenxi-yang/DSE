import torch
import time
import random
from torch.autograd import Variable
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import copy

from helper import *
from data_generator import *
from constants import *


def distance_f_point(pred_y, y):
    return torch.abs(pred_y.sub(y))


def distance_f_interval(X_list, target):
    alpha_smooth_max_var = var(alpha_smooth_max)
    res = var(0.0)
    res_up = var(0.0)
    res_base = var(0.0)

    if len(X_list) == 0:
        res = var(1.0)
        return res

    for X_table in X_list:
        X_min = X_table['x_min'].getInterval()
        X_max = X_table['x_max'].getInterval()
        pi = X_table['probability']
        p = X_table['explore_probability']

        X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
        X.left = torch.min(X_min.left, X_max.left)
        X.right = torch.max(X_min.right, X_max.right)

        reward = var(0.0)
        intersection_interval = get_intersection(X, target)
        
        if intersection_interval.isEmpty():
            reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
        
        tmp_res = reward.mul(pi)# pi.div(p))
        # tmp_res is the reward

        res = res.add(tmp_res)
    res = res.div(var(len(X_list)).add(EPSILON))
    # res = res_up.div(res_basse)
    return res


def distance_f_interval_REINFORCE(X_list, target, Theta):
    alpha_smooth_max_var = var(alpha_smooth_max)
    res = var(0.0)
    # print('X_list', len(X_list))
    reward_list = list()
    log_p_list = list()
    p_list = list()
    #! Smooth Max
    res_up = var(0.0)
    res_base = var(0.0)
    if len(X_list) == 0:
        res = var(1.0)
        return res
    for X_table in X_list:
        X_min = X_table['x_min'].getInterval()
        X_max = X_table['x_max'].getInterval()
        pi = X_table['probability']
        p = X_table['explore_probability']
        # print('pi, p', pi.data.item(), p.data.item())

        X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
        X.left = torch.min(X_min.left, X_max.left)
        X.right = torch.max(X_min.right, X_max.right)

        reward = var(0.0)
        intersection_interval = get_intersection(X, target)
        if intersection_interval.isEmpty():
            reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))

        tmp_res = reward.mul(pi.div(p))
        # tmp_res is the reward
        tmp_p = torch.log(pi)

        log_p_list.append(tmp_p)
        reward_list.append(reward)
        p_list.append(p)

        res = res.add(tmp_res)
    res = res.div(var(len(X_list)).add(EPSILON))
    
    return res, p_list, log_p_list, reward_list


def extract_result_safty(symbol_table_list):
    res_l, res_r = P_INFINITY, N_INFINITY
    for symbol_table in symbol_table_list:
        res_l = torch.min(res_l, symbol_table['x_min'].getInterval().left)
        res_r = torch.max(res_r, symbol_table['x_max'].getInterval().right)
    
    return res_l.data.item(), res_r.data.item()


def normal_pdf(x, mean, std):
    # print(x, mean, std)
    y = torch.exp((-((x-mean)**2)/(2*std*std)))/ (std* torch.sqrt(2*var(math.pi)))
    # print(y)
    # exit(0)
    return y


def generate_theta_sample_set(Theta):
    sample_theta_list = list()
    sample_theta_probability_list = list()
    for i in range(THETA_SAMPLE_SIZE):
        sample_theta = torch.normal(mean=Theta, std=var(1.0))
        sample_theta_probability = normal_pdf(sample_theta, Theta, var(1.0))
        # theta_normal.log_prob(sample_theta)
        # print(sample_theta, sample_theta_probability)
        sample_theta_list.append(sample_theta)
        sample_theta_probability_list.append(sample_theta_probability)
    return sample_theta_list, sample_theta_probability_list


def update_symbol_table_with_sample_theta(sample_theta_list, sample_theta_probability_list, symbol_table_list):
    for symbol_table in symbol_table_list:
        symbol_table['sample_theta'] = copy.deepcopy(sample_theta_list)
        symbol_table['sample_theta_probability'] = copy.deepcopy(sample_theta_probability_list)
    return symbol_table_list


def create_ball(x, width):
    res_l = list()
    res_r = list()
    for value in x:
        res_l.append(value-width)
        res_r.append(value+width)
    return res_l, res_r


def create_point_cloud(res_l, res_r, n=50):
    assert(len(res_l) == len(res_r))
    point_cloud = list()
    for i in range(n):
        point = list()
        for idx, v in enumerate(res_l):
            l = v
            r = res_r[idx]
            x = random.uniform(l, r)
            point.append(x)
        point_cloud.append(point)
    return point_cloud


def gd_direct_noise(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=0.01, epoch=1000, lr=0.00001, theta=None):
    print("--------------------------------------------------------------")
    print('----Gradient Direct Noise Descent Train DSE----')
    print('====Start Training====')
    len_theta = len(theta_l)
    TIME_OUT = False

    x_min = var(10000.0)
    x_max = var(0.0)
    x_smooth_min = var(10000.0)
    x_smooth_max = var(0.0)

    loop_list = list()
    loss_list = list()

    tmp_theta_list = [random.uniform(theta_l[idx], theta_r[idx]) for idx, value in enumerate(theta_l)]

    Theta = var_list(tmp_theta_list, requires_grad=True)

    root = construct_syntax_tree(Theta)
    root_smooth_point = construct_syntax_tree_smooth_point(Theta)
    root_point = construct_syntax_tree_point(Theta)

    # for 
    # for i in range(epoch):
    for width in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        num_partition = 0.0
        for idx, x in enumerate(X_train):
            x, y = x, y_train[idx]
            symbol_table_smooth_point = initialization_point(x)
            symbol_table_smooth_point = root_smooth_point['entry'].execute(symbol_table_smooth_point)
            data_loss = distance_f_point(symbol_table_smooth_point['res'], var(y))

            res_l, res_r = create_ball(x, width) # in the form of box
            point_cloud = create_point_cloud(res_l, res_r, n=50)
            symbol_table_list = initialization(res_l, res_r, point_cloud, y_train)
            symbol_table_list = root['entry'].execute(symbol_table_list)

            len_symbol_table = len(symbol_table_list)
            num_partition += len_symbol_table
            # print('# of partitions: ', len_symbol_table)
        print(f"width: {width}, over all path {num_partition * 1.0/len(X_train)}")


    return width



