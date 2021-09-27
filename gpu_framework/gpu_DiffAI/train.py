import copy
import random
import time
import torch
import constants 
import importlib

from utils import (
    batch_pair,
    divide_chunks,
    save_model,
    batch_pair_yield,
    )

import import_hub as hub
importlib.reload(hub)
from import_hub import *

random.seed(1)


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)
    return res_interval


def extract_safe_loss_old(component, target_component, target_idx):
    # component: {'trajectories', 'p_list'}

    if target_component["map_mode"] is False:
        safe_interval = target_component["condition"]
    # print(f"safe interval: {float(safe_interval.left)}, {float(safe_interval.right)}")
    method = target_component["method"]
    component_loss = var_list([0.0])
    min_l, max_r = 100000, -100000

    for trajectory in component['trajectories']:
        if method == "last":
            trajectory = [trajectory[-1]]
        elif method in ["all", "map_each"]:
            trajectory = trajectory
        else:
            raise NotImplementedError("Error: No trajectory method detected!")

        unsafe_penalty = var_list([0.0])
        for state_idx, state in enumerate(trajectory):
            X = domain.Interval(state[0][target_idx], state[1][target_idx])
            if target_component["map_mode"] is True:
                safe_interval = target_component["map_condition"][state_idx] # the constraint over the k-th step
            intersection_interval = get_intersection(X, safe_interval)
            if intersection_interval.isEmpty():
                if X.isPoint():
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right))
                else:
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength() + constants.eps)
            else:
                safe_portion = (intersection_interval.getLength() + eps).div(X.getLength() + eps)
                unsafe_value = 1 - safe_portion
            
            # sum
            unsafe_penalty = unsafe_penalty + unsafe_value
            # TODO: sum or max?
            # unsafe_penalty = torch.max(unsafe_penalty, unsafe_value)
            # print(f"X: {float(X.left), float(X.right)}, unsafe_value: {float(unsafe_value)}")
            min_l, max_r = min(min_l, float(X.left)), max(max_r, float(X.right))
        # exit(0)
        # TODO: max range increase, loss reduce?????
        component_loss += unsafe_penalty
    
    component_loss /= len(component['trajectories'])
    return component_loss, (min_l, max_r)


def extract_safe_loss(component, target_component, target_idx):
    # component: {'trajectories', 'p_list'}

    if target_component["map_mode"] is False:
        safe_interval_l, safe_interval_r = target_component["condition"].left, target_component["condition"].right
    else:
        safe_interval_list = target_component["map_condition"]

    trajectories_l = component['trajectories_l']
    trajectories_r = component['trajectories_r']
    converted_trajectories_l, converted_trajectories_r = list(zip(*trajectories_l)), list(zip(*trajectories_r))
    stacked_trajectories_l = [torch.stack(i) for i in converted_trajectories_l]
    stacked_trajectories_r = [torch.stack(i) for i in converted_trajectories_r]
    # exit(0)

    B, C = len(trajectories_l), len(converted_trajectories_l)
    unsafe_value = torch.zeros((B, C))
    if torch.cuda.is_available():
        unsafe_value = unsafe_value.cuda()
    min_l, max_r = 100000, -100000

    for state_idx, stacked_state in enumerate(stacked_trajectories_l):
        # print(stacked_state.shape, target_idx)
        pre_l = stacked_state[:, target_idx]
        pre_r = stacked_trajectories_r[state_idx][:, target_idx]
        if target_component['distance']:
            l = torch.zeros(pre_l.shape)
            r = torch.zeros(pre_r.shape)
            if torch.cuda.is_available():
                l = l.cuda()
                r = r.cuda()
            all_neg_index = torch.logical_and(pre_l<=0, pre_r<=0)
            across_index = torch.logical_and(pre_l<=0, pre_r>0)
            all_pos_index = torch.logical_and(pre_l>0, pre_r>0)
            l[all_neg_index], r[all_neg_index] = pre_r[all_neg_index].abs(), pre_l[all_neg_index].abs()
            l[across_index], r[across_index] = 0, pre_r[across_index]
            l[all_pos_index], r[all_pos_index] = pre_l[all_pos_index], pre_r[all_pos_index]
        else:
            l = pre_l
            r = pre_r
            
        if target_component["map_mode"] is True:
            safe_interval_sub_list = safe_interval_list[state_idx] 
            tmp_unsafe_value_list = list()
            for safe_interval in safe_interval_sub_list:
                tmp_unsafe_value = torch.zeros((B, 1))
                if torch.cuda.is_available():
                    tmp_unsafe_value = tmp_unsafe_value.cuda()
                safe_interval_l, safe_interval_r = safe_interval.left, safe_interval.right # the constraint over the k-th step
                intersection_l, intersection_r = torch.max(l, safe_interval_l), torch.min(r, safe_interval_r)
                empty_idx = intersection_r < intersection_l
                # other_idx = intersection_r >= intersection_l
                other_idx = ~ empty_idx
                # print(unsafe_value.shape, empty_idx.shape, state_idx)
                tmp_unsafe_value[empty_idx, 0] = torch.max(l[empty_idx] - safe_interval_r, safe_interval_l - r[empty_idx]) + 1.0
                tmp_unsafe_value[other_idx, 0] = 1 - (intersection_r[other_idx] - intersection_l[other_idx] + eps) / (r[other_idx] - l[other_idx] + eps)
                min_l, max_r = min(float(torch.min(l)), min_l), max(float(torch.max(r)), max_r)
                tmp_unsafe_value_list.append(tmp_unsafe_value)
            if len(tmp_unsafe_value_list) > 1:
                unsafe_value[:, state_idx] = torch.min(tmp_unsafe_value_list[0], tmp_unsafe_value_list[1]).squeeze()
            else:
                # print(unsafe_value[:, state_idx].shape, tmp_unsafe_value_list[0].shape, tmp_unsafe_value_list[0].squeeze().shape)
                unsafe_value[:, state_idx] = tmp_unsafe_value_list[0].squeeze()
        else:
            intersection_l, intersection_r = torch.max(l, safe_interval_l), torch.min(r, safe_interval_r)
            empty_idx = intersection_r < intersection_l
            # other_idx = intersection_r >= intersection_l
            other_idx = ~ empty_idx
            # print(unsafe_value.shape, empty_idx.shape, state_idx)
            unsafe_value[empty_idx, state_idx] = torch.max(l[empty_idx] - safe_interval_r, safe_interval_l - r[empty_idx]) + 1.0
            unsafe_value[other_idx, state_idx] = 1 - (intersection_r[other_idx] - intersection_l[other_idx] + eps) / (r[other_idx] - l[other_idx] + eps)
            min_l, max_r = min(float(torch.min(l)), min_l), max(float(torch.max(r)), max_r)

    unsafe_penalty = torch.max(unsafe_value, 1)[0]
    # print(unsafe_penalty.shape, torch.sum(unsafe_penalty))
    component_loss = torch.sum(unsafe_penalty)

    component_loss /= len(component['trajectories_l'])

    return component_loss, (min_l, max_r)
    

def safe_distance(component_result_list, target):
    # measure safe distance in DSE
    # take the average over components
    
    loss = var_list([0.0])
    min_l, max_r = 100000, -100000
    for idx, target_component in enumerate(target):
        target_loss = var_list([0.0])
        min_l, max_r = 100000, -100000
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

    expec_data_loss = var_list([0.0])
    count = 0
    for X, y in batch_pair_yield(trajectories, data_bs=512):
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        # print(f"after batch pair: {X.shape}")
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        yp = m(X, version="single_nn_learning")
        data_loss = criterion(yp, y)
        expec_data_loss += data_loss
        count += 1
    expec_data_loss = expec_data_loss / count
    return expec_data_loss


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

    # TODO: sample simultanuously
    # list of trajectories, p_list
    output_states = m(ini_states, 'abstract')

    safe_loss = safe_distance([output_states], target)
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
    print('====Start Training DiffAI====')

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
            if constants.profile:
                start_forward = time.time()
            data_loss = cal_data_loss(m, trajectories, criterion)
            # data_loss = var(0.0)
            safe_loss = cal_safe_loss(m, abstract_states, target)
            # safe_loss = var(1.0)

            print(f"data loss: {float(data_loss)}, safe loss: {float(safe_loss)}")
            
            loss = (data_loss + lambda_ * safe_loss) / lambda_
            if constants.profile:
                end_forward = time.time()
                print(f"--FORWARD: {end_forward - start_forward}")

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
                
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i - epochs_to_skip)}")
        print(f"-----finish {i}-th epoch-----, q: {float(data_loss)}, c: {float(safe_loss)}")
        if not constants.debug:
            log_file = open(constants.file_dir, 'a')
            log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i - epochs_to_skip)}\n")
            log_file.write(f"-----finish {i}-th epoch-----, q: {float(data_loss)}, c: {float(safe_loss)}\n")
            log_file.flush()

        if constants.early_stop is True:
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

        if (time.time() - start_time)/(i+1) > 3600*5 or TIME_OUT:
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
