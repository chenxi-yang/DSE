import copy
import random
import time

import matplotlib.pyplot as plt
import nlopt
import numpy as np
import torch
from constants 
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

def distance_f_point(pred_y, y):
    return torch.abs(pred_y.sub(y)) # l1-distance
    # return torch.square(pred_y.sub(y)) # l2-distance


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)
    return res_interval


def distance_f_interval(symbol_table_list, target):
    if len(symbol_table_list) == 0:
        return var(1.0)
    res = var(0.0)

    for symbol_table in symbol_table_list:
        X = symbol_table['safe_range']
        p = symbol_table['probability']

        # print(f"X: {X.left.data.item(), X.right.data.item()}, p: {p}")
        # print(f"target: {target.left, target.right}")
        intersection_interval = get_intersection(X, target)
        
        #  calculate the reward of each partition
        if intersection_interval.isEmpty():
            # print(f"in empty")
            reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            # print(f"not empty")
            # print(f"intersection interval get length: {intersection_interval.getLength()}")
            # print(f"X get length: {X.getLength()}")
            reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
        # print(f"reward for one partition: {reward}")
    
        tmp_res = reward.mul(p)
        res = res.add(tmp_res)
    res = res.div(var(len(symbol_table_list)).add(EPSILON))

    # print(f"safe loss, res: {res}")
    # res = res_up.div(res_basse)
    return res


def extract_safe_loss(component, target_component, target_idx):
    # component: {'trajectories', 'p_list'}

    safe_interval = target_component["condition"]
    # print(f"safe interval: {float(safe_interval.left)}, {float(safe_interval.right)}")
    method = target_component["method"]
    component_loss = var_list([0.0])

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
            unsafe_penalty = torch.max(unsafe_penalty, unsafe_value)
        component_loss += torch.log



    for symbol_table in abstract_state:
        if method == "last":
            trajectory = [symbol_table['trajectory'][-1]]
        elif method == "all":
            trajectory = symbol_table['trajectory'][:]
        else:
            raise NotImplementedError("Error: No trajectory method detected!")
        symbol_table['trajectory_loss'] = list()
        tmp_symbol_table_tra_loss = list()

        trajectory_loss = var_list([0.0])
        # print(f"symbol table p: {float(symbol_table['probability'])}")
        # print(f"start trajectory: ")
        for state in trajectory:
            # print(f"state: {state}")
            X = state[target_idx] # select the variable to measure
            if benchmark_name == "mountain_car_1":
                print(f"v: {float(state[0].left)}, {float(state[0].right)}; p: {float(state[1].left)}, {float(state[1].right)}; u: {float(state[2].left)}, {float(state[2].right)};")
            intersection_interval = get_intersection(X, safe_interval)
            if intersection_interval.isEmpty():
                # print(f"point: {X.isPoint()}")
                # print(f"empty")
                if X.isPoint():
                    # min point to interval
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right))
                else:
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength() + eps)
                # unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength().add(EPSILON))
            else:
                # print(f"not empty: {intersection_interval.getLength()}, {X.getLength()}")
                safe_portion = (intersection_interval.getLength() + eps).div(X.getLength() + eps)
                unsafe_value = 1 - safe_portion
            # if float(unsafe_value) > 0:
            # print(f"X: {float(X.left)}, {float(X.right)}")
            # print(f"unsafe value: {float(unsafe_value)}")
            # print(f"p: {symbol_table['probability']}")
                # print(f"point: {X.isPoint()}")
            # print(f"unsafe value: {float(unsafe_value)}")
            if outside_trajectory_loss:
                if benchmark_name in ["thermostat"]:
                    tmp_symbol_table_tra_loss.append(unsafe_value * symbol_table['probability'])
                else:
                    tmp_symbol_table_tra_loss.append(unsafe_value * symbol_table['probability']  / abstract_state_p)
            else:
                trajectory_loss = torch.max(trajectory_loss, unsafe_value)

        if outside_trajectory_loss:
            symbol_table_wise_loss_list.append(tmp_symbol_table_tra_loss)

        # print(f"add part: {trajectory_loss, symbol_table['probability']}")
        if not outside_trajectory_loss:
            if benchmark_name in ["thermostat"]:
                abstract_loss += trajectory_loss * symbol_table['probability']
            else:
                abstract_loss += trajectory_loss * symbol_table['probability'] / abstract_state_p
        # print(f"abstract_loss: {abstract_loss}")
    if outside_trajectory_loss:
        abstract_state_wise_trajectory_loss = zip(*symbol_table_wise_loss_list)
        abstract_loss = var_list([0.0])
        for l in abstract_state_wise_trajectory_loss:
            # print(l)
            abstract_loss = torch.max(abstract_loss, torch.sum(torch.stack(l)))
        # if debug:
        #     print(f"abstract_loss: {abstract_loss}")

    return abstract_loss
    

def safe_distance(component_result_list, target):
    # measure safe distance in DSE
    # take the average over components
    
    loss = var_list([0.0])
    p_list = list()
    for idx, target_component in enumerate(target):
        target_loss = var_list([0.0])
        # print(f"len abstract_state_list: {len(abstract_state_list)}")
        for component in component_result_list:
            component_safe_loss = extract_safe_loss(
                component, target_component, target_idx=idx, 
            )
            target_loss += component_safe_loss
        target_loss = target_loss / len(component_result_list)
        loss += target_loss
    return loss


def cal_data_loss(m, trajectories, criterion):
    # for the point in the same batch
    # calculate the data loss of each point
    # add the point data loss together
    if len(trajectories) == 0:
        return var_list([0.0])

    if constants.benchmark_name == 'thermostat':
        X, y = batch_pair_endpoint(trajectories, data_bs=None)
    else:
        X, y = batch_pair(trajectories, data_bs=512)
    # print(f"after batch pair: {X.shape}, {y.shape}")

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
            safe_loss = cal_safe_loss(m, abstract_states, target)

            print(f"data loss: {float(data_loss)}, safe loss: {float(safe_loss)}")
            
            loss = (data_loss + lambda_ * safe_loss) / lambda_

            loss.backward()
            print(f"value before clip, weight: {m.nn.linear1.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.detach().cpu().numpy().tolist()[0]}")
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
            print(f"grad before step, weight: {m.nn.linear1.weight.grad.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.grad.detach().cpu().numpy().tolist()[0]}")
            optimizer.step()
            print(f"value before step, weight: {m.nn.linear1.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.detach().cpu().numpy().tolist()[0]}")
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

        if (time.time() - start_time)/(i+1) > 900 or TIME_OUT:
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
