import torch
import torch.nn.functional as F
import torch.nn as nn

from random import shuffle

import domain
from constants import *
import constants

import math
import time
import copy

from utils import (
    show_cuda_memory,
    show_memory_snapshot,
)

import sys
import gc


'''
Module used as functions
'''

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        if not hasattr(self,'weight') or self.weight is None:
            return
        # print(f"weight size: {self.weight.size()}")
        # print(f"product: {product(self.weight.size())}")

        n = product(self.weight.size()) / self.out_channels
        stdv = 1 / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # print(f"weight: \n {self.weight}")
        # print(f"bias: \n {self.bias}")
        return x.matmul(self.weight).add(self.bias)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sigmoid()
  
    
class Tanh(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.tanh()


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.relu()


class SigmoidLinear(nn.Module):
    def __init__(self, sig_range):
        super().__init__()
        self.sig_range = sig_range
    
    def forward(self, x):
        return x.sigmoid_linear(sig_range=self.sig_range)

'''
Program Statement
'''

def smooth_join_trajectory(trajectory_1, trajectory_2, alpha_prime_1, alpha_prime_2, alpha_1, alpha_2):
    l1, l2 = len(trajectory_1), len(trajectory_2)
    trajectory = list()
    K = min(l1, l2)
    for idx in range(K):
        states_1, states_2 = trajectory_1[0], trajectory_2[0]
        l_s = len(states_1)
        state_list = list()
        for state_idx in range(l_s):
            state_1, state_2 = states_1[0], states_2[0]
            a = state_1.smoothJoin(state_2, alpha_prime_1, alpha_prime_2, alpha_1, alpha_2)
            state_list.append(a)
        trajectory.append(state_list)
    # print(f"middle: {len(trajectory), l1, l2}")
    
    if l1 < l2:
        trajectory.extend(trajectory_2[l1:])
    elif l1 > l2:
        trajectory.extend(trajectory_1[l2:])
    # print(len(trajectory), l1, l2)
    assert(len(trajectory) == max(l1, l2))

    return trajectory


def update_joined_tables(res_states, new_c, new_delta, new_trajectory, new_idx, new_p, new_alpha):
    # print(f"smooth join, c:{new_c}, delta: {new_delta}")
    if 'x' in res_states:
        res_states['x'].c = torch.cat((res_states['x'].c, new_c), 0)
        res_states['x'].delta = torch.cat((res_states['x'].delta, new_delta), 0)
        res_states['trajectories'].append(new_trajectory)
        res_states['idx_list'].append(new_idx)
        res_states['p_list'].append(new_p)
        res_states['alpha_list'].append(new_alpha)
    else:
        res_states['x'] = domain.Box(new_c, new_delta)
        res_states['trajectories'] = [new_trajectory]
        res_states['idx_list'] = [new_idx]
        res_states['p_list'] = [new_p]
        res_states['alpha_list'] = [new_alpha]

    return res_states


def smooth_join(states1, states2):
    # symbol_tables
    # 'x': B*D, 'trajectories': trajectory of each B, 'idx_list': idx of B in order
    if len(states1) == 0:
        return states2
    if len(states2) == 0:
        return states1

    res_states = dict()
    idx1, idx2 = 0, 0
    idx_list_1, idx_list_2 = states1['idx_list'], states2['idx_list']
    p_list_1, p_list_2 = states1['p_list'], states2['p_list']
    alpha_list_1, alpha_list_2 = states1['alpha_list'], states2['alpha_list']

    while idx1 <= len(idx_list_1) - 1 or idx2 <= len(idx_list_2) - 1:
        if idx1 > len(idx_list_1) - 1 or (idx2 <= len(idx_list_2) - 1 and idx_list_1[idx1] > idx_list_2[idx2]):
            new_c = states2['x'].c[idx2:idx2+1].clone()
            new_delta = states2['x'].delta[idx2:idx2+1].clone()
            new_trajectory = states2['trajectories'][idx2]
            new_idx = states2['idx_list'][idx2]
            new_p = states2['p_list'][idx2]
            new_alpha = states2['alpha_list'][idx2]
            res_states = update_joined_tables(res_states, new_c, new_delta, new_trajectory, new_idx, new_p, new_alpha)
            idx2 += 1
        elif idx2 > len(idx_list_2) - 1 or (idx1 <= len(idx_list_1) - 1 and idx_list_1[idx1] < idx_list_2[idx2]):
            new_c = states1['x'].c[idx1:idx1+1].clone()
            new_delta = states1['x'].delta[idx1:idx1+1].clone()
            new_trajectory = states1['trajectories'][idx1]
            new_idx = states1['idx_list'][idx1]
            new_p = states1['p_list'][idx1]
            new_alpha = states1['alpha_list'][idx1]
            res_states = update_joined_tables(res_states, new_c, new_delta, new_trajectory, new_idx, new_p, new_alpha)
            idx1 += 1
        else: # idx_list_1[idx_1] == idx_list_2[idx_2], need to join
            assert(idx_list_1[idx1] == idx_list_2[idx2])
            # if 1 and 2 have the same idx, do the join
            # convert to domain-prime
            
            alpha_max = torch.max(states1['alpha_list'][idx1], states2['alpha_list'][idx2])
            assert(float(alpha_max) > 0.0)
            alpha_prime_1, alpha_prime_2 = states1['alpha_list'][idx1] / alpha_max, states2['alpha_list'][idx2] / alpha_max

            c_out = (states1['alpha_list'][idx1] * states1['x'].c[idx1:idx1+1] + states2['alpha_list'][idx2] * states2['x'].c[idx2:idx2+1]) / (states1['alpha_list'][idx1] + states2['alpha_list'][idx2])
            new_c_1, new_c_2 = alpha_prime_1 * states1['x'].c[idx1:idx1+1] + (1 - alpha_prime_1) * c_out, alpha_prime_2 * states2['x'].c[idx2:idx2+1] + (1 - alpha_prime_2) * c_out
            new_delta_1, new_delta_2 = alpha_prime_1 * states1['x'].delta[idx1:idx1+1], alpha_prime_2 * states2['x'].c[idx2:idx2+1]
            
            new_left = torch.min(new_c_1 - new_delta_1, new_c_2 - new_delta_2)
            new_right = torch.max(new_c_1 + new_delta_1, new_c_2 + new_delta_2)
            new_c = (new_left + new_right) / 2.0
            new_delta = (new_right - new_left) / 2.0
            new_trajectory = smooth_join_trajectory(states1['trajectories'][idx1], states2['trajectories'][idx2], alpha_prime_1, alpha_prime_2, alpha_1=states1['alpha_list'][idx1], alpha_2=states2['alpha_list'][idx2])
            new_idx = idx_list_1[idx1]
            new_p = p_list_1[idx1]
            new_alpha = torch.min(var(1.0), states1['alpha_list'][idx1] + states2['alpha_list'][idx2])

            res_states = update_joined_tables(res_states, new_c, new_delta, new_trajectory, new_idx, new_p, new_alpha)
            idx2 += 1
            idx1 += 1

    return res_states


def calculate_states(target_idx, arg_idx, f, states):
    x = states['x']
    # print(f"x: {x.c, x.delta}")
    input = x.select_from_index(1, arg_idx)
    res = f(input)
    # print(f)
    # print(f"calculate, input: {input.c, input.delta}, res: {res.c, res.delta}")
    # TODO: check
    # print(f'cal')
    # print(f)
    # print(x.c.shape, x.delta.shape)
    # print(target_idx)
    # print(res.c.shape, res.c.shape)
    x.c[:, target_idx] = res.c 
    x.delta[:, target_idx] = res.delta
    states['x'] = x
    return states


def extract_branch_alpha(target, test):
    alpha_test = torch.zeros(target.getLeft().shape).cuda()
    alpha_test[target.getRight() <= test] = 1.0
    alpha_test[target.getLeft() > test] = 0.0
    cross_idx = torch.logical_and(target.getRight() > test, target.getLeft() <= test)

    # update with a smooth coefficient
    alpha_test[cross_idx] = torch.min(var(1.0), (test - target.getLeft()[cross_idx]) / ((target.getRight()[cross_idx] - target.getLeft()[cross_idx]).mul(constants.INTERVAL_BETA)))
    # print(f"alpha_test: {alpha_test}")
    return alpha_test, 1 - alpha_test


def calculate_branch(target_idx, test, states):
    body_states, orelse_states = dict(), dict()
    x = states['x']
    target = x.select_from_index(1, target_idx) # select the batch target from x

    # select the idx of left = target.left < test,  right = target.right >= test
    # update alpha based on the volume
    # select the trajectory accordingly
    # select the idx accordingly
    # split the other
    alpha_left, alpha_right = extract_branch_alpha(target, test)
    left, right = alpha_left > 0, alpha_right > 0
    # print(f"left: {left.tolist()}, right: {right.tolist()}")

    if True in left: # split to left
        left_idx = left.nonzero(as_tuple=True)[0].tolist()
        x_left = domain.Box(x.c[left.squeeze(1)], x.delta[left.squeeze(1)])
        left_target_c, left_target_delta = target.c[left].unsqueeze(1), target.delta[left].unsqueeze(1)
        # get the new c, delta
        new_left_target_c = ((left_target_c - left_target_delta) + torch.min((left_target_c + left_target_delta), test)) / 2.0
        new_left_target_delta = (torch.min((left_target_c + left_target_delta), test) - (left_target_c - left_target_delta)) / 2.0
        x_left.c[:, target_idx:target_idx+1] = new_left_target_c
        x_left.delta[:, target_idx:target_idx+1] = new_left_target_delta

        body_states['x'] = x_left
        body_states['trajectories'] = [states['trajectories'][i] for i in left_idx]
        body_states['idx_list'] = [states['idx_list'][i] for i in left_idx]
        body_states['p_list'] = [states['p_list'][i] for i in left_idx]
        body_states['alpha_list'] = [states['alpha_list'][i].mul(alpha_left[i]) for i in left_idx]
        # print(f"body idx: {body_states['idx_list']}")
        # print(f"body_states: {body_states['alpha_list'].tolist()}")
    
    if True in right: # split to right
        right_idx = right.nonzero(as_tuple=True)[0].tolist()
        x_right = domain.Box(x.c[right.squeeze(1)], x.delta[right.squeeze(1)])
        right_target_c, right_target_delta = target.c[right].unsqueeze(1), target.delta[right].unsqueeze(1)

        new_right_target_c = (torch.max((right_target_c - right_target_delta), test) + (right_target_c + right_target_delta)) / 2.0
        new_right_target_delta = ((right_target_c + right_target_delta) - torch.max((right_target_c - right_target_delta), test)) / 2.0
        x_right.c[:, target_idx:target_idx+1] = new_right_target_c
        x_right.delta[:, target_idx:target_idx+1] = new_right_target_delta

        orelse_states['x'] = x_right
        orelse_states['trajectories'] = [states['trajectories'][i] for i in right_idx]
        orelse_states['idx_list'] = [states['idx_list'][i] for i in right_idx]
        orelse_states['p_list'] = [states['p_list'][i] for i in right_idx]
        orelse_states['alpha_list'] = [states['alpha_list'][i].mul(alpha_right[i]) for i in right_idx]
    
    return body_states, orelse_states


class Skip(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, states):
        return states


class Assign(nn.Module):
    def __init__(self, target_idx, arg_idx: list(), f):
        super().__init__()
        self.f = f
        self.target_idx = torch.tensor(target_idx)
        self.arg_idx = torch.tensor(arg_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
            self.arg_idx = self.arg_idx.cuda()
    
    def forward(self, states):
        # TODO: update
        res_states = calculate_states(self.target_idx, self.arg_idx, self.f, states)

        return res_states


class IfElse(nn.Module):
    def __init__(self, target_idx, test, f_test, body, orelse):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.f_test = f_test
        self.body = body
        self.orelse = orelse
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, states):
        test = self.f_test(self.test)
        body_states, orelse_states = calculate_branch(self.target_idx, self.test, states)
        if len(body_states) > 0:
            body_states = self.body(body_states)
        if len(orelse_states) > 0:
            orelse_states = self.orelse(orelse_states)
        
        # maintain the same number of components as the initial ones
        # TODO: update sound join
        res_states = smooth_join(body_states, orelse_states)

        return res_states


class While(nn.Module):
    def __init__(self, target_idx, test, body):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.body = body
        if torch.cuda.is_available():
            # print(f"CHECK: cuda")
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, states):
        i = 0
        res_states = list()
        while(len(states) > 0):
            body_states, orelse_states = calculate_branch(self.target_idx, self.test, states)
            # TODO: update
            res_states = smooth_join(res_states, orelse_states)
            if len(body_states) == 0:
                return res_states
            states = self.body(body_states)
            i += 1
            if i > MAXIMUM_ITERATION:
                break
        res_states = smooth_join(res_states, orelse_states)
        res_states = smooth_join(res_states, body_states)

        return res_states


class Trajectory(nn.Module):
    # TODO: update, add state in trajectory list
    def __init__(self, target_idx):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, states):
        x = states['x']
        trajectories = states['trajectories']
        B, D = x.c.shape
        for x_idx in range(B):
            cur_x_c, cur_x_delta = x.c[x_idx], x.delta[x_idx]
            input_interval_list = list()
            for idx in self.target_idx:
                input = domain.Box(cur_x_c[idx], cur_x_delta[idx])
                input_interval = input.getInterval()
                # print(f"interval: {float(input_interval.left), float(input_interval.right)}")
                assert float(input_interval.left) <= float(input_interval.right)
                input_interval_list.append(input_interval)
            trajectories[x_idx].append(input_interval_list)
        
        states['trajectories'] = trajectories

        return states




            

            









