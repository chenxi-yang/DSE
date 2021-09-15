import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import constants
from torch.autograd import Variable
import importlib

from utils import (
    batch_pair,
    divide_chunks,
    save_model,
    batch_pair_trajectory,
    )

import import_hub as hub
importlib.reload(hub)
from import_hub import *

random.seed(1)

def cal_data_loss(m, trajectories, criterion):
    # for the point in the same batch
    # calculate the data loss of each point
    # add the point data loss together
    if len(trajectories) == 0:
        return var_list([0.0])
    # print('only data')
    if constants.benchmark_name in ['thermostat']:
        X, y_trajectory = batch_pair_trajectory(trajectories, data_bs=None, standard_value=70.0)
        X, y_trajectory = torch.from_numpy(X).float(), [torch.from_numpy(y).float() for y in y_trajectory]
        # print(f"after batch pair: {X.shape}")
        if torch.cuda.is_available():
            X, y_trajectory = X.cuda(), [y.cuda() for y in y_trajectory]
        yp_trajectory = m(X, version="single_nn_learning")
        data_loss = var(0.0)
        for idx, yp in enumerate(yp_trajectory):
            # print(yp.shape, y_trajectory[idx].shape)
            data_loss = data_loss + criterion(yp, y_trajectory[idx])
            data_loss /= len(yp_trajectory)
    else:
        X, y = batch_pair(trajectories, data_bs=512)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        # print(f"after batch pair: {X.shape}")
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        yp = m(X, version="single_nn_learning")
        data_loss = criterion(yp, y)
    
    if constants.debug:
        yp_list = yp.squeeze().detach().cpu().numpy().tolist()
        y_list = y.squeeze().detach().cpu().numpy().tolist()
        print(f"yp: {yp_list[:5]}, {min(yp_list)}, {max(yp_list)}")
    
        # print(f"x: {X}")
        yp_list = yp.squeeze().detach().cpu().numpy().tolist()
        y_list = y.squeeze().detach().cpu().numpy().tolist()

        print(f"yp: {min(yp_list)}, {max(yp_list)}")
        print(f"y: {min(y_list)}, {max(y_list)}")
        print(data_loss)
        exit(0)
    
    return data_loss


def learning(
        m, 
        components,
        lambda_=None, 
        epoch=1000,
        target=None, 
        lr=0.00001,
        bs=10,
        nn_mode='all',
        l=10,
        save=None,
        epochs_to_skip=None,
        model_name=None,
        data_bs=None,
        ):
    print("--------------------------------------------------------------")
    print('====Start Training only data====')

    TIME_OUT = False
    torch.autograd.set_detect_anomaly(True)

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

        epoch_data_loss = 0.0
        batch_count = 0
        for trajectories, abstract_states in divide_chunks(components, bs=bs, data_bs=data_bs):

            data_loss = cal_data_loss(m, trajectories, criterion)

            # print(f"data loss: {float(data_loss)}")
            epoch_data_loss += float(data_loss)
            batch_count += 1
            
            loss = data_loss

            loss.backward()
            # print(f"value before clip, weight: {m.nn.linear1.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.detach().cpu().numpy().tolist()[0]}")
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
            # print(f"grad before step, weight: {m.nn.linear1.weight.grad.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.grad.detach().cpu().numpy().tolist()[0]}")
            optimizer.step()
            # print(f"value before step, weight: {m.nn.linear1.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.detach().cpu().numpy().tolist()[0]}")
            optimizer.zero_grad()
        
        if save:
            save_model(m, constants.MODEL_PATH, name=model_name, epoch=i)
            print(f"save model")
                
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1 - epochs_to_skip)}")
        print(f"-----finish {i}-th epoch-----, q: {epoch_data_loss/batch_count}")
        if not constants.debug:
            log_file = open(constants.file_dir, 'a')
            log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}\n")
            log_file.write(f"-----finish {i}-th epoch-----, q: {epoch_data_loss/batch_count}\n")
            log_file.flush()

        if (time.time() - start_time)/(i+1) > 900 or TIME_OUT:
            if not constants.debug:
                log_file = open(constants.file_dir, 'a')
                log_file.write('TIMEOUT: avg epoch time > 3600sec \n')
                log_file.close()
            TIME_OUT = True
            break

    if not constants.debug:
        log_file = open(constants.file_dir, 'a')
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
