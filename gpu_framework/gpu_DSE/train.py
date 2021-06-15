import copy
import random
import time

import matplotlib.pyplot as plt
import nlopt
import numpy as np
import torch
from constants import benchmark_name
from torch.autograd import Variable

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


def extract_abstract_state_safe_loss(abstract_state, target_component, target_idx):
    # weighted sum of symbol_table loss in one abstract_state
    unsafe_probability_condition = target_component["phi"]
    safe_interval = target_component["condition"]
    # print(f"safe interval: {float(safe_interval.left)}, {float(safe_interval.right)}")
    method = target_component["method"]
    abstract_loss = var_list([0.0])
    symbol_table_wise_loss_list = list()
    abstract_state_p = var_list([0.0])
    for symbol_table in abstract_state:
        abstract_state_p += symbol_table['probability']

    # print(f"in each abstract state: {len(abstract_state)}")
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
    

# TODO: to change
def safe_distance(abstract_state_list, target):
    # measure safe distance in DSE
    # I am using sampling, and many samples the eventual average will be the same as the expectation
    # limited number of abstract states, so count sequentially based on target is ok
    
    loss = var_list([0.0])
    p_list = list()
    for idx, target_component in enumerate(target):
        target_loss = var_list([0.0])
        # print(f"len abstract_state_list: {len(abstract_state_list)}")
        for abstract_state in abstract_state_list:
            abstract_state_safe_loss = extract_abstract_state_safe_loss(
                abstract_state, target_component, target_idx=idx, 
            )
            target_loss += abstract_state_safe_loss
        target_loss = target_loss / (len(abstract_state_list) + eps)
        # Weighted loss of different state variables
        # print(f"target_loss: {float(target_loss)}")
        target_loss = target_component["w"] * (target_loss - target_component['phi'])
        # print(f"refined target_loss: {float(target_loss)}")
        # TODO: max(loss - target, 0) change or not
        # target_loss = torch.max(target_loss, var(0.0))
        if 'fairness' in benchmark_name:
            p_list.append(target_loss)
        else:
            loss += target_loss
    
    if 'fairness' in benchmark_name:
        # lower bound
        p_h_f = 1 - torch.max(p_list[0], var(0.01))
        p_m = 1 - torch.max(p_list[3], var(0.01))
        # upper bound
        p_h_m = torch.max(p_list[1], var(0.01))
        p_f = torch.max(p_list[2], var(0.01))

        # >= 0.9
        lower_bound_ratio = (p_h_f * p_m) / (p_h_m * p_f)
        print(p_list)
        loss = 1 - (p_h_f * p_m) / (p_h_m * p_f)
    # print(f"loss: {loss}")
    # exit(0)
    # TODO: add the part for computation across target loss

    return loss


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
    assert(len(ini_abstract_state_list) == 1)
    res_states = list()

    # TODO: sample simultanuously
    for i in range(constants.SAMPLE_SIZE):
        output_states = m(ini_states, 'abstract')
        res_states.append(output_states)

    safe_loss = safe_distance(res_states, target)
    return safe_loss


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
            save_model(m, MODEL_PATH, name=model_name, epoch=i)
            print(f"save model")
                
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1 - epochs_to_skip)}")
        print(f"-----finish {i}-th epoch-----, q: {float(data_loss)}, c: {float(safe_loss)}")
        if not debug:
            log_file = open(file_dir, 'a')
            log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}\n")
            log_file.write(f"-----finish {i}-th epoch-----, q: {float(data_loss)}, c: {float(safe_loss)}\n")
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
    # q = var(0.0)

    data_loss = cal_data_loss(m, X_train, y_train)

    # for idx, x in enumerate(X_train):
    #     x, y = x, y_train[idx]
    #     symbol_table_point = initialization_point(x)
    #     symbol_table_point = root_point['entry'].execute(symbol_table_point)

    #     # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
    #     q = q.add(distance_f_point(symbol_table_point['res'], var(y)))

    # q = q.div(var(len(X_train)))
    print(f"cal_q, {data_loss}")
    
    return q
