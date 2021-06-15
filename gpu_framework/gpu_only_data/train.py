import copy
import random
import time

import matplotlib.pyplot as plt
import nlopt
import numpy as np
import torch
from constants import benchmark_name
from torch.autograd import Variable

# define the model by the same structure
from gpu_DSE.modules import *

from utils import (
    batch_pair,
    batch_pair_endpoint,
    ini_trajectory,
    show_component, 
    show_cuda_memory,
    show_trajectory,
    divide_chunks,
    )

random.seed(1)


def cal_data_loss(m, trajectories, criterion):
    # for the point in the same batch
    # calculate the data loss of each point
    # add the point data loss together
    if len(trajectories) == 0:
        return var_list([0.0])

    X, y = batch_pair(trajectory_list, data_bs=512)
    print(f"after batch pair: {X.shape}, {y.shape}")

    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()
    
    yp = m(X, version="single_nn_learning")
    if debug:
        yp_list = yp.squeeze().detach().cpu().numpy().tolist()
        y_list = y.squeeze().detach().cpu().numpy().tolist()
        print(f"yp: {yp_list[:5]}, {min(yp_list)}, {max(yp_list)}")
    
    # print(f"x: {X}")
    yp_list = yp.squeeze().detach().cpu().numpy().tolist()
    y_list = y.squeeze().detach().cpu().numpy().tolist()

    print(f"yp: {min(yp_list)}, {max(yp_list)}")
    print(f"y: {min(y_list)}, {max(y_list)}")
    data_loss = criterion(yp, y)
    
    return data_loss


def learning(
        m, 
        components,
        lambda_=lambda_, 
        epoch=1000,
        target=None, 
        lr=0.00001,
        bs=10,
        nn_mode='all',
        l=10,
        save=save,
        epochs_to_skip=None,
        model_name=None,
        data_bs=data_bs,
        ):
    print("--------------------------------------------------------------")
    print('====Start Training only data====')

    TIME_OUT = False

    if torch.cuda.is_available():
        m.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-06)

    if epochs_to_skip is None:
        epochs_to_skip = -1

    start_time = time.time()

    for i in range(epoch):
        if i <= epochs_to_skip:
            continue

        for trajectories, abstract_states in divide_chunks(components, bs=bs, data_bs=None):

            data_loss = cal_data_loss(m, trajectories, criterion)

            print(f"data loss: {float(data_loss)}")
            
            loss = data_loss

            loss.backward()
            print(f"value before clip, weight: {m.nn.linear1.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.detach().cpu().numpy().tolist()[0]}")
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
            print(f"grad before step, weight: {m.nn.linear1.weight.grad.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.grad.detach().cpu().numpy().tolist()[0]}")
            optimizer.step()
            print(f"value before step, weight: {m.nn.linear1.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.detach().cpu().numpy().tolist()[0]}")
            optimizer.zero_grad()
        
        if save:
            save_model(m, MODEL_PATH, name=model_name, epoch=i)
            print(f"save model")
                
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1 - epochs_to_skip)}")
        print(f"-----finish {i}-th epoch-----, q: {float(data_loss)}")
        if not debug:
            log_file = open(file_dir, 'a')
            log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}\n")
            log_file.write(f"-----finish {i}-th epoch-----, q: {float(data_loss)}\n")
            log_file.flush()

        if (time.time() - start_time)/(i+1) > 900 or TIME_OUT:
            if not debug:
                log_file = open(file_dir, 'a')
                log_file.write('TIMEOUT: avg epoch time > 3600sec \n')
                log_file.close()
            TIME_OUT = True
            break

    if not debug:
        log_file = open(file_dir, 'a')
        spend_time = time.time() - start_time
        log_file.write(f"One train: Optimization-- ' + total time: {spend_time}, total epochs: {i + 1}, avg time: {spend_time/(i + 1)}\n")
        log_file.close()
    
    return [], 0.0, [], 0.0, 0.0, TIME_OUT


def cal_c(X_train, y_train, m, target):
    # TODO: to check the cal_c process
    # only for calculating the value instead of the gradient
    print(f"---in cal_c---")
    # print(f"theta, {theta}")
    c_loss = var(0.0)
    for idx, x in enumerate(X_train):
        x, y = x, y_train[idx]
        loss = var(0.0)
        safe_loss = cal_safe_loss(m, x, width, target)
        c_loss += safe_loss
    c = c_loss.div(len(X_train))
    print(f"---cal_c, {c}")

    return c


def cal_q(X_train, y_train, m):
    root_point = construct_syntax_tree_point(theta)

    data_loss = cal_data_loss(m, X_train, y_train)
    print(f"cal_q, {data_loss}")
    
    return q
