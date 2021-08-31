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
    aggregate_sampling_states,
    )

import import_hub as hub
importlib.reload(hub)
from import_hub import *

random.seed(1)

def get_intersection(l, r, l1, r1):
    res_interval = domain.Interval()
    res_interval.left = torch.max(l, l1)
    res_interval.right = torch.min(r, r1)
    return res_interval


def extract_safe_loss_old(component, target_component, target_idx):
    # component: {'trajectories', 'p_list'}

    if target_component["map_mode"] is False:
        safe_interval = target_component["condition"]
    # print(f"safe interval: {float(safe_interval.left)}, {float(safe_interval.right)}")
    method = target_component["method"]
    component_loss = var_list([0.0])
    real_safety_loss = 0.0
    min_l, max_r = 100000, -100000

    for trajectory, p in zip(component['trajectories'], component['p_list']):
        if method == "last":
            trajectory = [trajectory[-1]]
        elif method in ["all", "map_each"]:
            trajectory = trajectory
        else:
            raise NotImplementedError("Error: No trajectory method detected!")

        unsafe_penalty = var_list([0.0])
        p0_left_list, p1_left_list, p2_left_list = list(), list(), list()
        a_list, b_list, c_list = list(), list(), list()
        x_left_list,  x_right_list, unsafe_value_list = list(), list(), list()
        for state_idx, state in enumerate(trajectory):
            # X = state[target_idx]
            l, r = state[0][target_idx], state[1][target_idx]
            # print(f"trajectory length: {len(trajectory)}")
            # print(state_idx)
            if target_component["map_mode"] is True:
                safe_interval = target_component["map_condition"][state_idx] # the constraint over the k-th step
            intersection = get_intersection(l, r, safe_interval)
            if intersection.isEmpty():
                # update safe loss
                unsafe_value = torch.max(l.sub(safe_interval.right), safe_interval.left.sub(r))
                unsafe_value = unsafe_value + 1.0
                # if X.isPoint():
                #     unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right))
                # else:
                #     unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength() + constants.eps)
            else:
                safe_portion = (intersection.getLength() + eps).div((r - l) + eps)
                unsafe_value = 1 - safe_portion
            
            # use sum
            # try use the sum to penalize
            # unsafe_penalty = unsafe_penalty + unsafe_value
            unsafe_penalty = torch.max(unsafe_penalty, unsafe_value)

            # print(f"x1: {float(state[1].left), float(state[1].right)}, y1: {float(state[2].left), float(state[2].right)}")
            # print(f"x2: {float(state[3].left), float(state[3].right)}, y2: {float(state[4].left), float(state[4].right)}")
            # print(f"p0: {float(state[5].left), float(state[5].right)}, p1: {float(state[6].left), float(state[6].right)}")
            # print(f"p2: {float(state[7].left), float(state[7].right)}, p3: {float(state[8].left), float(state[8].right)}")
            
            # race_track
            # print(f"p0: {float(state[2].left), float(state[2].right)}, p1: {float(state[3].left), float(state[3].right)}")
            # print(f"p2: {float(state[4].left), float(state[4].right)}")
            # print(f"X: {float(X.left), float(X.right)}, unsafe_value: {float(unsafe_value)}")
            # if float(X.left) == 0.0:
            #     exit(0)
            # p0_left_list.append(float(state[2].left))
            # p1_left_list.append(float(state[3].left))
            # p2_left_list.append(float(state[4].left))
            # a_list.append((float(state[5].left), float(state[5].right)))
            # b_list.append((float(state[6].left), float(state[6].right)))
            # c_list.append((float(state[7].left), float(state[7].right)))
            # x_left_list.append(float(X.left))
            # x_right_list.append(float(X.right))
            # unsafe_value_list.append(float(unsafe_value))
            # if state_idx == len(trajectory) - 1:
            #     # print(f"p0: {float(state[2].left), float(state[2].right)}, p1: {float(state[3].left), float(state[3].right)}")
            #     # print(f"p2: {float(state[4].left), float(state[4].right)}")
            #     print(f"last step X: {float(X.left), float(X.right)}, unsafe_value: {float(unsafe_value)}")
            min_l, max_r = min(min_l, float(l)), max(max_r, float(r))
        # print(len(trajectory))
        # print(f"p0: {p0_left_list}")
        # print(f"p1: {p1_left_list}")
        # print(f"p2: {p2_left_list}")
        # print(f"a: {a_list}\nb: {b_list}\nc: {c_list}")
        # print(f"x left: {x_left_list}")
        # print(f"x right: {x_right_list}")
        # print(f"unsafe_value list: {unsafe_value_list}")
        # print(f"p: {float(p)}, unsafe_penalty: {float(unsafe_penalty)}")
        # exit(0)
        component_loss += p * float(unsafe_penalty) + unsafe_penalty
        real_safety_loss += float(unsafe_penalty)
    
    component_loss /= len(component['p_list'])
    real_safety_loss /= len(component['p_list'])

    return component_loss, real_safety_loss, (min_l, max_r)


def extract_safe_loss(component, target_component, target_idx):
    # component: {'trajectories', 'p_list'}

    if target_component["map_mode"] is False:
        safe_interval_l, safe_interval_r = target_component["condition"].left, target_component["condition"].right
    else:
        safe_interval_list = target_component["map_condition"]
    # print(f"safe interval: {float(safe_interval.left)}, {float(safe_interval.right)}")
    # method = target_component["method"]
    # component_loss = 0.0
    # real_safety_loss = 0.0
    # min_l, max_r = 100000, -100000
    # trajectories have the same length
    # print(len(component['trajectories']))

    trajectories_l = component['trajectories_l']
    trajectories_r = component['trajectories_r']
    p_list = torch.stack(component['p_list'])
    converted_trajectories_l, converted_trajectories_r = list(zip(*trajectories_l)), list(zip(*trajectories_r))
    stacked_trajectories_l = [torch.stack(i) for i in converted_trajectories_l]
    stacked_trajectories_r = [torch.stack(i) for i in converted_trajectories_r]

    B, C = len(trajectories_l), len(converted_trajectories_l)
    unsafe_value = torch.zeros((B, C))
    if torch.cuda.is_available():
        unsafe_value = unsafe_value.cuda()
    min_l, max_r = 100000, -100000

    for state_idx, stacked_state in enumerate(stacked_trajectories_l):
        # print(stacked_state.shape, target_idx)
        l = stacked_state[:, target_idx]
        r = stacked_trajectories_r[state_idx][:, target_idx]
        if target_component["map_mode"] is True:
            safe_interval_l, safe_interval_r = safe_interval_list[state_idx].left, safe_interval_list[state_idx].right # the constraint over the k-th step
        
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
    sum_penalty = torch.sum(unsafe_penalty)
    # !!! detach!!!
    # print(p_list.shape, unsafe_penalty.shape)
    component_loss = torch.dot(p_list.squeeze(), unsafe_penalty.detach()) + sum_penalty
    real_safety_loss = float(sum_penalty)

    component_loss /= len(component['p_list'])
    real_safety_loss /= len(component['p_list'])

    return component_loss, real_safety_loss, (min_l, max_r)

    # for trajectory, p in zip(component['trajectories'], component['p_list']):
    #     # if constants.profile:
    #     #     start_trajectory = time.time()
    #     if method == "last":
    #         trajectory = [trajectory[-1]]
    #     elif method in ["all", "map_each"]:
    #         trajectory = trajectory
    #     else:
    #         raise NotImplementedError("Error: No trajectory method detected!")

    #     unsafe_penalty = 0.0
    #     for state_idx, state in enumerate(trajectory):
    #         # X = state[target_idx]
    #         l, r = state[0][target_idx], state[1][target_idx]
    #         if target_component["map_mode"] is True:
    #             safe_interval_l, safe_interval_r = safe_interval_list[state_idx].left,  safe_interval_list[state_idx].right # the constraint over the k-th step
            
    #         # if constants.profile:
    #         #     start_unsafe_penalty = time.time()
    #         intersection_l, intersection_r = max(l, safe_interval_l), min(r, safe_interval_r)
    #         # if constants.profile:
    #         #     end_intersection = time.time()
    #         #     print(f"--INTERSECTION: {end_intersection - start_unsafe_penalty}")
    #         if intersection_r < intersection_l:
    #             # update safe loss
    #             unsafe_value = max(l - (safe_interval_r), safe_interval_l - (r)) + 1.0
    #         else:
    #             safe_portion = (intersection_r - intersection_l + eps) / ((r - l) + eps)
    #             unsafe_value = 1 - safe_portion
    #         # if constants.profile:
    #         #     end_calculation = time.time()
    #             # print(f"--CALCULATION: {end_calculation - end_intersection}")
    #         unsafe_penalty = max(unsafe_penalty, unsafe_value)
    #         # if constants.profile:
    #         #     end_unsafe_penalty = time.time()
    #             # print(f"---UNSAFE PENALTY: {end_unsafe_penalty - end_calculation}")
    #         min_l, max_r = min(min_l, float(l)), max(max_r, float(r))

    #     component_loss += p * float(unsafe_penalty) + unsafe_penalty
    #     real_safety_loss += float(unsafe_penalty)
    #     # if constants.profile:
    #     #     end_trajectory = time.time()
    #         # print(f"--ONE TRAJECTORY: {end_trajectory - start_trajectory}")
    #         # exit(0)
    
    # component_loss /= len(component['p_list'])
    # real_safety_loss /= len(component['p_list'])

    # return component_loss, real_safety_loss, (min_l, max_r)


def safe_distance(component_result_list, target):
    # measure safe distance in DSE
    # take the average over components
    
    loss = var_list([0.0])
    real_safety_loss = 0.0
    min_l, max_r = 100000, -100000
    for idx, target_component in enumerate(target):
        target_loss = var_list([0.0])
        real_target_loss = 0.0
        # print(f"len abstract_state_list: {len(abstract_state_list)}")
        for component in component_result_list:
            component_safe_loss, real_safety_loss, (tmp_min_l, tmp_max_r) = extract_safe_loss(
                component, target_component, target_idx=idx, 
            )
            target_loss += component_safe_loss
            real_target_loss += real_safety_loss
            min_l, max_r = min(min_l, tmp_min_l), max(max_r, tmp_max_r)
        target_loss = target_loss / len(component_result_list)
        real_target_loss = real_target_loss / len(component_result_list)
        loss += target_loss
        real_safety_loss += real_target_loss
    print(f"range of trajectory: {min_l, max_r}")
    if not constants.debug:
        log_file = open(constants.file_dir, 'a')
        log_file.write(f"range of trajectory: {min_l, max_r}\n")
        log_file.flush()
    return loss, real_safety_loss


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
        # print(f"finish data loss")
        # exit(0)
    
    # if constants.debug:
    # yp_list = yp.squeeze().detach().cpu().numpy().tolist()
    # y_list = y.squeeze().detach().cpu().numpy().tolist()
    # print(f"yp: {yp_list[:5]}, {min(yp_list)}, {max(yp_list)}")

    # # print(f"x: {X}")
    # yp_list = yp.squeeze().detach().cpu().numpy().tolist()
    # y_list = y.squeeze().detach().cpu().numpy().tolist()

    # print(f"yp: {min(yp_list)}, {max(yp_list)}")
    # print(f"y: {min(y_list)}, {max(y_list)}")
    
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
            'trajectories_l': list(),
            'trajectories_r': list(),
            'p_list': list(),
        }
        component_result_list.append(component)
    
    # TODO: sample simultanuously
    # aggregate abstract states based on sample_size
    aggregated_abstract_states_list = aggregate_sampling_states(abstract_states, constants.SAMPLE_SIZE)
    if constants.profile:
        start = time.time()
    for aggregated_abstract_states in aggregated_abstract_states_list:
        ini_states = initialize_components(aggregated_abstract_states)
        # print(f"start safe AI")
        output_states = m(ini_states, 'abstract')
        trajectories_l = output_states['trajectories_l']
        trajectories_r = output_states['trajectories_r']
        idx_list = output_states['idx_list']
        p_list = output_states['p_list']

        ziped_result = zip(idx_list, trajectories_l, trajectories_r, p_list)
        sample_result = [(x, y, z, i) for x, y, z, i in sorted(ziped_result, key=lambda tuple: tuple[0])]
        for idx, trajectory_l, trajectory_r, p in sample_result:
            component_result_list[0]['trajectories_l'].append(trajectory_l)
            component_result_list[0]['trajectories_r'].append(trajectory_r)
            component_result_list[0]['p_list'].append(p)
    if constants.profile:
        end = time.time()
        print(f"--SAFE OUTPUT EXTRACTION: {end - start}")
        start_safety_loss_calculation = time.time()
    safe_loss, real_safety_loss = safe_distance([component_result_list[0]], target)
    if constants.profile:
        end_safety_loss_calculation = time.time()
        print(f"--SAFETY LOSS CALCULATION: {end_safety_loss_calculation - start_safety_loss_calculation}")
    
    # list of trajectories, p_list
    # for i in range(constants.SAMPLE_SIZE):
    #     ini_states = initialize_components(abstract_states)
    #     # print(f"start safe AI")
    #     output_states = m(ini_states, 'abstract')
    #     trajectories = output_states['trajectories']
    #     idx_list = output_states['idx_list']
    #     p_list = output_states['p_list']

    #     ziped_result = zip(idx_list, trajectories, p_list)
    #     sample_result = [(x, y, z) for x, y, z in sorted(ziped_result, key=lambda tuple: tuple[0])]
    #     for idx, trajectory, p in sample_result:
    #         component_result_list[idx]['trajectories'].append(trajectory)
    #         component_result_list[idx]['p_list'].append(p)
        # exit(0)

    # safe_loss, real_safety_loss = safe_distance(component_result_list, target)
    return safe_loss, real_safety_loss


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
            # if constants.profile:
            #     start_forward = time.time()
            data_loss = cal_data_loss(m, trajectories, criterion)
            # if constants.profile:
            #     end_data_loss = time.time()
            #     print(f"DATA LOSS: {end_data_loss - start_forward}")
            # data_loss = var(0.0)
            safe_loss, real_safety_loss = cal_safe_loss(m, abstract_states, target)
            # safe_loss = var(1.0)

            print(f"data loss: {float(data_loss)}, safe loss: {float(safe_loss)}, real_safety_loss: {real_safety_loss}")
            loss = (data_loss + lambda_ * safe_loss) / lambda_
            # if constants.profile:
            #     end_forward = time.time()
            #     print(f"--FORWARD: {end_forward - start_forward}")

            # if constants.profile:
            #     start_sgd = time.time()
            loss.backward(retain_graph=True)
            # if constants.profile:
            #     end_sgd = time.time()
            #     print(f"--BACK SGD: {end_sgd - start_sgd}")
            # print(f"value before clip, weight: {m.nn.linear1.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.detach().cpu().numpy().tolist()[0]}")
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
            # print(f"grad before step, weight: {m.nn.linear1.weight.grad.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.grad.detach().cpu().numpy().tolist()[0]}")
            # if constants.profile:
            #     start_step = time.time()
            optimizer.step()
            # print(f"value before step, weight: {m.nn.linear1.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear1.bias.detach().cpu().numpy().tolist()[0]}")
            optimizer.zero_grad()
            # if constants.profile:
            #     end_step = time.time()
            #     print(f"--OPTIMIZATION STEP: {end_step - start_step}")
        
        if save:
            save_model(m, constants.MODEL_PATH, name=model_name, epoch=i)
            print(f"save model")
                
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i - epochs_to_skip)}")
        print(f"-----finish {i}-th epoch-----, q: {float(data_loss)}, c: {float(safe_loss)}, real_c: {real_safety_loss}")
        # if constants.profile:
        #     exit(0)
        if not constants.debug:
            log_file = open(constants.file_dir, 'a')
            log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i - epochs_to_skip)}\n")
            log_file.write(f"-----finish {i}-th epoch-----, q: {float(data_loss)}, c: {float(safe_loss)}, real_c: {real_safety_loss}\n")
            log_file.flush()
        
        if constants.early_stop is True:
            if abs(float(safe_loss)) <= float(EPSILON):
                end_count += 1
            else:
                end_count = 0
            
            if end_count >= 5: # the signal to early stop
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
