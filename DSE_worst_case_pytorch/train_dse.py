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


def distance_f_interval_new(X_list, target):
    for X_table in X_list:
        X_min = X_table['x_min'].getInterval()
        X_max = X_table['x_max'].getInterval()
        res = var(0.0)

        X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
        X.left = torch.min(X_min.left, X_max.left)
        X.right = torch.max(X_min.right, X_max.right)

        reward = var(0.0)
        intersection_interval = get_intersection(X, target)
        if intersection_interval.isEmpty():
            # print('isempty')
            reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            # print('not empty')
            reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
        
        res = torch.max(res, reward)
    # print(f"result length, {len(X_list)}, reward: {res}")
    # res is the worst case cost of X_list
    return res


def extract_result_safty(symbol_table_list):
    res_l, res_r = P_INFINITY, N_INFINITY
    for symbol_table in symbol_table_list:
        res_l = torch.min(res_l, symbol_table['x_min'].getInterval().left)
        res_r = torch.max(res_r, symbol_table['x_max'].getInterval().right)
    
    return res_l.data.item(), res_r.data.item()


def normal_pdf(x, mean, std):
    # print(x, mean, std)
    y = torch.exp((-((x-mean)**2)/(2*std*std)))/ (std* torch.sqrt(2*var(math.pi)))
    res = var(1.0)
    # multiple all in y
    for p in y:
        res = res.mul(p)

    # print(f"pdf, {y}, {res}")
    # exit(0)
    res = res.mul(10)
    return res


def generate_theta_sample_set(Theta):
    # sample_theta_list = list()
    # sample_theta_probability_list = list()
    # print(f"----------generating Theta --------------")
    # print(f"Original theta: {Theta}")
    sample_theta_list = list()
    for i in range(THETA_SAMPLE_SIZE):
        sample_theta = torch.normal(mean=Theta, std=var(1.0))
        # print(f"Sampled theta: {Theta}")
        # sample_theta = torch.distributions.multivariate_normal.MultivariateNormal(loc=Theta, torch.eye(1.0))
        sample_theta_probability = normal_pdf(sample_theta, Theta, var(1.0))
        sample_theta_list.append((sample_theta, sample_theta_probability))
    # print(f"---------finish generating Theta----------")
    return sample_theta_list


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


def cal_data_loss(theta, x, y):
    # print(f"------- point start-----")
    # start_time = time.time()
    root_point = construct_syntax_tree_point(theta)
    # point_time = time.time()
    # print(f"point time: {point_time - start_time}")
    symbol_table_point = initialization_point(x)
    # ini_time = time.time()
    # print(f"ini time: {ini_time - point_time}")
    symbol_table_point = root_point['entry'].execute(symbol_table_point)
    # run_time = time.time()
    # print(f"run time: {run_time - ini_time}")
    data_loss = distance_f_point(symbol_table_point['res'], var(y))
    # print(f"------- point finish -----")
    return data_loss


def cal_safe_loss(theta, x, width):
    # print(f"------- SL start-----")
    # start_time = time.time()
    root = construct_syntax_tree(theta)
    # sl_time = time.time()
    # print(f"SL time: {sl_time - start_time}")
    res_l, res_r = create_ball(x, width)
    symbol_table_list = initialization(res_l, res_r, [], [])
    # ini_time = time.time()
    # print(f"ini time: {ini_time - sl_time}")
    symbol_table_list = root['entry'].execute(symbol_table_list)
    l, r = extract_result_safty(symbol_table_list)
    # run_time = time.time()
    # print(f"run time: {run_time - ini_time}")
    safe_loss = distance_f_interval_new(symbol_table_list, target)
    # print(f"------- SL end-----")
    # print(f"l, r, safe loss: {l, r, safe_loss}")

    return safe_loss


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

    theta_training_time = 0.0
    start_time = time.time()
    for i in range(epoch):
        num_partition = 0.0
        cur_time = time.time()
        # data_loss_total = var(0.0)
        # safe_loss_total = var(0.0)
        f_loss = var(0.0)
        q_loss = var(0.0)
        c_loss = var(0.0)

        q_loss_wo_p = var(0.0)
        c_loss_wo_p = var(0.0)
        for idx, x in enumerate(X_train):
            data_time = time.time()
            x, y = x, y_train[idx]
            sample_theta_list = generate_theta_sample_set(Theta)
            real_data_loss = var(0.0)
            real_safe_loss = var(0.0)
            loss = var(0.0)
            tmp_q_loss_wo_p = var(0.0)
            tmp_c_loss_wo_p = var(0.0)
            for (sample_theta, sample_theta_p) in sample_theta_list:
                theta_time = time.time()
                data_loss = cal_data_loss(sample_theta, x, y)
                safe_loss = cal_safe_loss(sample_theta, x, width)
                res = data_loss + lambda_.mul(safe_loss)
                
                real_data_loss += data_loss * sample_theta_p
                real_safe_loss += safe_loss * sample_theta_p
                # data_loss_total = data_loss_total.add(data_loss)
                # safe_loss_total = safe_loss_total.add(safe_loss)
                tmp_q_loss_wo_p += data_loss
                tmp_c_loss_wo_p += safe_loss

                loss += var(res.data.item()).mul(var(sample_theta_p.data.item())).mul(torch.log(sample_theta_p))
                theta_training_time += time.time() - theta_time
            q_loss += real_data_loss
            c_loss += real_safe_loss

            q_loss_wo_p += tmp_q_loss_wo_p.div(len(sample_theta_list))
            c_loss_wo_p += tmp_c_loss_wo_p.div(len(sample_theta_list))
            # print(f"average training time for one theta, {theta_training_time/len(sample_theta_list)}")
            loss.backward(retain_graph=True)
            
            with torch.no_grad():
                # print(f"grad:{Theta.grad}")
                for theta_idx in range(len_theta):
                    try:
                        # print(noise)
                        # Theta[theta_idx].data -= lr * (dTheta[theta_idx].data + var(random.uniform(-noise, noise)))
                        Theta[theta_idx].data -= lr * (Theta.grad[theta_idx] + var(random.uniform(-noise, noise)))
                        
                    except RuntimeError: # for the case no gradient with Theta[theta_idx]
                        Theta[theta_idx].data -= lr * (var(random.uniform(-noise, noise)))
                Theta.grad.zero_()
            
            for theta_idx in range(len_theta):
                if Theta[theta_idx].data.item() <= theta_l[theta_idx] or Theta[theta_idx].data.item() >= theta_r[theta_idx]:
                    Theta[theta_idx].data.fill_(random.uniform(theta_l[theta_idx], theta_r[theta_idx]))
            
            # print(f"training time for one data point, {time.time() - data_time}")
            # print(f"real data loss:{real_data_loss}, real safe loss:{real_safe_loss}, probability")
            if i > 30:
                lr *= 0.5
        
        f_loss = q_loss + lambda_ * c_loss
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}")
        print(f"-----finish {i}-th epoch-----, q: {q_loss.data.item()}, c: {c_loss.data.item()}")
        print(f"------{i}-th epoch------, avg q: {q_loss_wo_p.div(len(X_train))}, avg c: {c_loss_wo_p.div(len(X_train))}")
        # if torch.abs(f_loss.data) < var(stop_val):
        #     break
        
        if (time.time() - start_time)/(i+1) > 300:
            log_file = open(file_dir, 'a')
            log_file.write('TIMEOUT: avg epoch time > 300s \n')
            log_file.close()
            TIME_OUT = True
            break
    
    res = f_loss.div(len(X_train))

    log_file = open(file_dir, 'a')
    spend_time = time.time() - start_time
    log_file.write('Optimization:' + str(spend_time) + ',' + str(i+1) + ',' + str(spend_time/(i+1)) + '\n')
    log_file.close()
    
    return Theta, res, [], data_loss, safe_loss, TIME_OUT


def cal_c(X_train, y_train, theta):
    # only for calculating the value instead of the gradient
    print(f"---in cal_c---")
    print(f"theta, {theta}")
    c_loss = var(0.0)
    for idx, x in enumerate(X_train):
        x, y = x, y_train[idx]
        loss = var(0.0)
        safe_loss = cal_safe_loss(theta, x, width)
        c_loss += safe_loss
    c = c_loss.div(len(X_train))
    print(f"---cal_c, {theta}, {c}")

    return c


def cal_q(X_train, y_train, theta):
    root_point = construct_syntax_tree_point(theta)
    q = var(0.0)

    for idx, x in enumerate(X_train):
        x, y = x, y_train[idx]
        symbol_table_point = initialization_point(x)
        symbol_table_point = root_point['entry'].execute(symbol_table_point)

        # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
        q = q.add(distance_f_point(symbol_table_point['res'], var(y)))

    q = q.div(var(len(X_train)))
    print(f"cal_q, {theta}, {q}")
    
    return q
