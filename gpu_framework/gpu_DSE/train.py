import copy
import random
import time

import matplotlib.pyplot as plt
import nlopt
import numpy as np
import torch
import constants 
from torch.autograd import Variable
import importlib

from utils import (
    batch_pair,
    batch_pair_endpoint,
    ini_trajectory,
    show_component, 
    show_cuda_memory,
    show_trajectory,
    divide_chunks,
    save_model,
    load_model,
    import_benchmarks,
    )
from domain_utils import (
    concatenate_states,
)

if constants.benchmark_name == "thermostat":
    import benchmarks.thermostat as tm
    importlib.reload(tm)
    from benchmarks.thermostat import *
elif constants.benchmark_name == "mountain_car":
    import benchmarks.mountain_car as mc
    importlib.reload(mc)
    from benchmarks.mountain_car import *
elif constants.benchmark_name == "unsmooth_1":
    from benchmarks.unsmooth import *
elif constants.benchmark_name == "unsmooth_2_separate":
    from benchmarks.unsmooth_2_separate import *
elif constants.benchmark_name == "unsmooth_2_overall":
    from benchmarks.unsmooth_2_overall import *
elif constants.benchmark_name == "path_explosion":
    from benchmarks.path_explosion import *
elif constants.benchmark_name == "path_explosion_2":
    from benchmarks.path_explosion_2 import *

random.seed(1)


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)
    return res_interval


def extract_safe_loss(component, target_component, target_idx):
    # component: {'trajectories', 'p_list'}

    safe_interval = target_component["condition"]
    # print(f"safe interval: {float(safe_interval.left)}, {float(safe_interval.right)}")
    method = target_component["method"]
    component_loss = var_list([0.0])
    min_l, max_r = 100000, -100000

    for trajectory, p in zip(component['trajectories'], component['p_list']):
        if method == "last":
            trajectory = [trajectory[-1]]
        elif method == "all":
            trajectory = trajectory
        else:
            raise NotImplementedError("Error: No trajectory method detected!")

        unsafe_penalty = var_list([0.0])
        for state in trajectory:
            X = state[target_idx]
            intersection_interval = get_intersection(X, safe_interval)
            if intersection_interval.isEmpty():
                if X.isPoint():
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right))
                else:
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength() + constants.eps)
            else:
                safe_portion = (intersection_interval.getLength() + eps).div(X.getLength() + eps)
                unsafe_value = 1 - safe_portion
            # unsafe_penalty = unsafe_penalty + unsafe_value
            unsafe_penalty = torch.max(unsafe_penalty, unsafe_value)
            # print(f"X: {float(X.left), float(X.right)}, unsafe_value: {float(unsafe_value)}")
            min_l, max_r = min(min_l, float(X.left)), max(max_r, float(X.right))
        # print(f"p: {p}, unsafe_penalty: {unsafe_penalty}")
        component_loss += p * float(unsafe_penalty) + unsafe_penalty
    
    component_loss /= len(component['p_list'])
    return component_loss, (min_l, max_r)
    

def safe_distance(component_result_list, target):
    # measure safe distance in DSE
    # take the average over components
    
    loss = var_list([0.0])
    p_list = list()
    min_l, max_r = 100000, -100000
    for idx, target_component in enumerate(target):
        target_loss = var_list([0.0])
        # print(f"len abstract_state_list: {len(abstract_state_list)}")
        for component in component_result_list:
            component_safe_loss, (tmp_min_l, tmp_max_r) = extract_safe_loss(
                component, target_component, target_idx=idx, 
            )
            target_loss += component_safe_loss
            min_l, max_r = min(min_l, tmp_min_l), max(max_r, tmp_max_r)
        target_loss = target_loss / len(component_result_list)
        loss += target_loss
    print(f"range of trajectory: {min_l, max_r}")
    if not constants.debug:
        log_file = open(constants.file_dir, 'a')
        log_file.write(f"range of trajectory: {min_l, max_r}\n")
        log_file.flush()
    return loss


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
        print(f"after batch pair: {X.shape}")
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
    
    return data_loss


def cal_safe_loss(m, abstract_states, target):
    '''
    DSE: sample paths
    abstract_state = list<{
        'center': vector, 
        'width': vector, 
    }>
    '''
    # show_component(abstract_state)
    ini_states = initialize_components(abstract_states)
    component_result_list = list()
    for i in range(len(ini_states['idx_list'])):
        component = {
            'trajectories': list(),
            'p_list': list(),
        }
        component_result_list.append(component)

    # TODO: sample simultanuously
    # list of trajectories, p_list
    for i in range(constants.SAMPLE_SIZE):
        output_states = m(ini_states, 'abstract')
        trajectories = output_states['trajectories']
        idx_list = output_states['idx_list']
        p_list = output_states['p_list']

        ziped_result = zip(idx_list, trajectories, p_list)
        sample_result = [(x, y, z) for x, y, z in sorted(ziped_result, key=lambda tuple: tuple[0])]
        for idx, trajectory, p in sample_result:
            component_result_list[idx]['trajectories'].append(trajectory)
            component_result_list[idx]['p_list'].append(p)

    safe_loss = safe_distance(component_result_list, target)
    return safe_loss


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
    print('====Start Training DSE====')

    TIME_OUT = False
    end_count = 0

    if torch.cuda.is_available():
        m.cuda()
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-06)

    if epochs_to_skip is None:
        epochs_to_skip = -1

    start_time = time.time()

    print(f"epochs_to_skip: {epochs_to_skip}")

    for i in range(epoch):
        if i <= epochs_to_skip:
            continue

        for trajectories, abstract_states in divide_chunks(components, bs=bs, data_bs=None):
            if constants.run_time_debug:
                data_loss_time = time.time()
            data_loss = cal_data_loss(m, trajectories, criterion)
            # data_loss = var(0.0)
            safe_loss = cal_safe_loss(m, abstract_states, target)

            print(f"data loss: {float(data_loss)}, safe loss: {float(safe_loss)}")
            
            loss = (data_loss + lambda_ * safe_loss) / lambda_

            loss.backward(retain_graph=True)
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
        print(f"-----finish {i}-th epoch-----, q: {float(data_loss)}, c: {float(safe_loss)}")
        if not constants.debug:
            log_file = open(constants.file_dir, 'a')
            log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}\n")
            log_file.write(f"-----finish {i}-th epoch-----, q: {float(data_loss)}, c: {float(safe_loss)}\n")
            log_file.flush()
        
        if float(safe_loss) == 0.0:
            end_count += 1
        else:
            end_count = 0
        if end_count >= 3:
            if not constants.debug:
                log_file = open(file_dir, 'a')
                log_file.write('EARLY STOP: Get safe results \n')
                log_file.close()
            break

        if (time.time() - start_time)/(i+1) > 3600 or TIME_OUT:
            if not constants.debug:
                log_file = open(file_dir, 'a')
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
