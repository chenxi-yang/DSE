import torch
import torch.nn.functional as F
import torch.nn as nn

from random import shuffle

import domain
from helper import * 
from constants import *
import constants

import math
import time


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

# def calculate_x_list(target_idx, arg_idx, f, symbol_tables):
#     # assign_time = time.time()
#     for idx, symbol_table in enumerate(symbol_table_list):
#         x = symbol_table['x']
#         input = x.select_from_index(1, arg_idx) # torch.index_select(x, 0, arg_idx)
#         # print(f"f: {f}")
#         res = f(input)
#         # print(f"calculate_x_list --  target_idx: {target_idx}, res: {res.c}, {res.delta}")
#         x.set_from_index(target_idx, res) # x[target_idx[0]] = res
        
#         symbol_table['x'] = x
#         # symbol_table['probability'] = symbol_table['probability'].mul(p)
#         symbol_table_list[idx] = symbol_table
#     # print(f"-- assign -- calculate_x_list: {time.time() - assign_time}")
#     return symbol_table_list

def calculate_x_list(target_idx, arg_idx, f, symbol_tables):
    x = symbol_tables['x']
    input = x.select_from_index(1, arg_idx)
    res = f(input)
    x.c[:, target_idx] = res.c 
    x.delta[:, target_idx] = res.delta
    symbol_tables['x'] = x

    return symbol_tables
        

def calculate_branch(target_idx, test, symbol_tables):
    body_symbol_tables, orelse_symbol_tables = dict(), dict()
    x = symbol_tables['x']
    target = x.select_from_index(1, target_idx) # select the batch target from x

    # select the idx of left = target.right < test,  right = target.right >= test
    # select the trajectory accordingly
    # select the idx accordingly
    # split the other

    left = target.getLeft() < test
    if True in left: # split to left
        left_idx = left.nonzero(as_tuple=True)[0].tolist()
        # x_left.c, x_left.delta = x.c[left.squeeze()], x.delta[left.squeeze()]
        x_left = domain.Box(x.c[left.squeeze()], x.delta[left.squeeze()])
        left_target_c, left_target_delta = target.c[left].unsqueeze(1), target.delta[left].unsqueeze(1)

        # get the new c, delta
        left_target_c = ((left_target_c - left_target_delta) + torch.min((left_target_c + left_target_delta), test)) / 2.0
        left_target_delta = (torch.min((left_target_c + left_target_delta), test) - (left_target_c - left_target_delta)) / 2.0
        x_left.c[:, target_idx:target_idx+1] = left_target_c
        x_left.delta[:, target_idx:target_idx+1] = left_target_delta
        body_symbol_tables['x'] = x_left
        body_symbol_tables['trajectory_list'] = [symbol_tables['trajectory_list'][i] for i in left_idx]
        body_symbol_tables['idx_list'] = [symbol_tables['idx_list'][i] for i in left_idx]
    
    right = target.getRight() >= test
    if True in right: # split to right
        right_idx = right.nonzero(as_tuple=True)[0].tolist()
        x_right = domain.Box(x.c[right.squeeze()], x.delta[right.squeeze()])
        right_target_c, right_target_delta = target.c[right].unsqueeze(1), target.delta[right].unsqueeze(1)

        right_target_c = (torch.max((right_target_c - right_target_delta), test) + (right_target_c + right_target_delta)) / 2.0
        right_target_delta = ((right_target_c + right_target_delta) - torch.max((right_target_c - right_target_delta), test)) / 2.0
        x_right.c[:, target_idx:target_idx+1] = right_target_c
        x_right.delta[:, target_idx:target_idx+1] = right_target_delta
        orelse_symbol_tables['x'] = x_right
        orelse_symbol_tables['trajectory_list'] = [symbol_tables['trajectory_list'][i] for i in right_idx]
        orelse_symbol_tables['idx_list'] = [symbol_tables['idx_list'][i] for i in right_idx]

    return body_symbol_tables, orelse_symbol_tables


def sound_join_trajectory(trajectory_1, trajectory_2):
    l1, l2 = len(trajectory_1), len(trajectory_2)
    trajectory = list()
    for idx in range(min(l1, l2)):
        states_1, states_2 =  trajectory_1[idx], trajectory_2[idx]
        state_list = list()
        for state_idx, v in enumerate(states_1):
            state_1, state_2 = states_1[state_idx], states_2[state_idx]
            state_list.append(state_1.soundJoin(state_2))
        trajectory.append(state_list)
    if l1 < l2:
        trajectory.extend(trajectory_2[l1:])
    elif l1 > l2:
        trajectory.extend(trajectory_1[l2:])
    
    return trajectory


def update_joined_tables(res_symbol_tables, new_c, new_delta, new_trajectory, new_idx):
    if 'x' in res_symbol_tables:
        res_symbol_tables['x'].c = torch.cat((res_symbol_tables['x'].c, new_c), 0)
        res_symbol_tables['x'].delta = torch.cat((res_symbol_tables['x'].delta, new_delta), 0)
        res_symbol_tables['trajectory_list'].append(new_trajectory)
        res_symbol_tables['idx_list'].append(new_idx)
    else:
        res_symbol_tables['x'] = domain.Box(new_c, new_delta)
        res_symbol_tables['trajectory_list'] = [new_trajectory]
        res_symbol_tables['idx_list'] = [new_idx]

    return res_symbol_tables


def sound_join(symbol_tables_1, symbol_tables_2):
    # symbol_tables
    # 'x': B*D, 'trajectory_list': trajectory of each B, 'idx_list': idx of B in order
    if len(symbol_tables_1) == 0:
        return symbol_tables_2
    if len(symbol_tables_2) == 0:
        return symbol_tables_1

    res_symbol_tables = dict()
    idx1, idx2 = 0, 0
    idx_list_1, idx_list_2 = symbol_tables_1['idx_list'], symbol_tables_2['idx_list']
    while idx1 <= len(idx_list_1) - 1 or idx2 <= len(idx_list_2) - 1:
        if idx1 > len(idx_list_1) - 1 or idx_list_1[idx1] > idx_list_2[idx2]:
            new_c = symbol_tables_2['x'].c[idx2:idx2+1]
            new_delta = symbol_tables_2['x'].delta[idx2:idx2+1]
            new_trajectory = symbol_tables_2['trajectory_list'][idx2]
            new_idx =  symbol_tables_2['idx_list'][idx2]
            res_symbol_tables = update_joined_tables(res_symbol_tables, new_c, new_delta, new_trajectory, new_idx)
            idx2 += 1
        elif idx2 > len(idx_list_2) - 1 or idx_list_1[idx1] < idx_list_2[idx2]:
            new_c = symbol_tables_1['x'].c[idx1:idx1+1]
            new_delta = symbol_tables_1['x'].delta[idx1:idx1+1]
            new_trajectory = symbol_tables_1['trajectory_list'][idx1]
            new_idx =  symbol_tables_1['idx_list'][idx1]
            res_symbol_tables = update_joined_tables(res_symbol_tables, new_c, new_delta, new_trajectory, new_idx)
            idx1 += 1
        else: # idx_list_1[idx_1] == idx_list_2[idx_2], need to join
            assert(idx_list_1[idx_1] == idx_list_2[idx_2])
            new_left = torch.min(symbol_tables_1['x'].c[idx1:idx1+1] - symbol_tables_1['x'].delta[idx1:idx1+1], symbol_tables_2['x'].c[idx2:idx2+1] - symbol_tables_2['x'].delta[idx2:idx2+1])
            new_right = torch.max(symbol_tables_1['x'].c[idx1:idx1+1] + symbol_tables_1['x'].delta[idx1:idx1+1], symbol_tables_2['x'].c[idx2:idx2+1] + symbol_tables_2['x'].delta[idx2:idx2+1])
            new_c = (new_left + new_right) / 2.0
            new_delta = (new_right - new_left) / 2.0
            new_trajectory = sound_join_trajectory(symbol_tables_1['trajectory_list'][idx1], symbol_tables_2['trajectory_list'][idx2])
            new_idx = idx_list_1[idx_1]
            res_symbol_tables = update_joined_tables(res_symbol_tables, new_c, new_delta, new_trajectory, new_idx)
            idx2 += 1
            idx1 += 1

    return res_symbol_tables


class Skip(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, symbol_tables, cur_sample_size=0):
        return symbol_tables


class Assign(nn.Module):
    def __init__(self, target_idx, arg_idx: list(), f):
        super().__init__()
        self.f = f
        self.target_idx = torch.tensor(target_idx)
        self.arg_idx = torch.tensor(arg_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
            self.arg_idx = self.arg_idx.cuda()
    
    def forward(self, symbol_tables, cur_sample_size=0):
        # print(f"Assign Before: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        res_symbol_tables = calculate_x_list(self.target_idx, self.arg_idx, self.f, symbol_tables)
        # print(f"Assign After: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        return res_symbol_tables


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
    
    def forward(self, symbol_tables):
        test = self.f_test(self.test)

        body_symbol_tables, orelse_symbol_tables = calculate_branch(self.target_idx, self.test, symbol_tables)
        # print(f"{len(branch_list)}")
        
        # print(f"In IfElse, {len(body_list)}, {len(else_list)}")
        
        if len(body_symbol_tables) > 0:
            body_symbol_tables = self.body(body_symbol_tables)
        if len(orelse_symbol_tables) > 0:
            orelse_symbol_tables = self.orelse(orelse_symbol_tables)
        
        res_symbol_tables = sound_join(body_symbol_tables, orelse_symbol_tables)

        return res_symbol_tables


class While(nn.Module):
    def __init__(self, target_idx, test, body):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.body = body
        if torch.cuda.is_available():
            # print(f"CHECK: cuda")
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, symbol_tables):
        '''
        super set of E_{i-th step} and [\neg condition]
        '''
        # symbol_tables are
        # print(f"##############In while DiffAI#########")
        i = 0
        res_symbol_tables = dict()
        while(len(symbol_tables) > 0):
            # counter += 1
            body_symbol_tables, orelse_symbol_tables = calculate_branch(self.target_idx, self.test, symbol_tables)

            res_symbol_tables = sound_join(res_symbol_tables, orelse_symbol_tables)

            if len(body_symbol_tables) == 0:
                return res_symbol_tables
            
            symbol_tables = self.body(body_symbol_tables)

            i += 1
            if i > 500:
                print(f"Exceed maximum iterations: Have to END.")
                break
        res_symbol_tables = sound_join(res_symbol_tables, orelse_symbol_tables)
        return res_symbol_tables


# def update_trajectory(symbol_table, target_idx):
#     input_interval_list = list()
#     # print(f"all symbol_table: {symbol_table['x'].c, symbol_table['x'].delta}")
#     for idx in target_idx:
#         input = symbol_table['x'].select_from_index(0, idx)
#         input_interval = input.getInterval()
#         # print(f"idx:{idx}, input: {input.c, input.delta}")
#         # print(f"input_interval: {input_interval.left.data.item(), input_interval.right.data.item()}")
#         assert input_interval.left.data.item() <= input_interval.right.data.item()
#         input_interval_list.append(input_interval)
#     # print(f"In update trajectory")
#     symbol_table['trajectory'].append(input_interval_list)
#     # exit(0)
#     # print(f"Finish update trajectory")

#     return symbol_table


class Trajectory(nn.Module):
    def __init__(self, target_idx):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, symbol_tables, cur_sample_size=0):
        x = symbol_tables['x']
        trajectory_list = symbol_tables['trajectory_list']
        
        B, D = x.c.shape
        for x_idx in range(B):
            cur_x_c, cur_x_delta = x.c[x_idx], x.delta[x_idx]
            input_interval_list = list()
            for idx in target_idx:
                input = domain.Box(cur_x_c[idx], cur_x_delta[idx])
                input_interval = input.getInterval()
                assert input_interval.left.data.item() <= input_interval.right.data.item()
                input_interval_list.append(input_interval)
            trajectory_list[x_idx].append(input_interval_list)
        
        symbol_tables['trajectory_list'] = trajectory_list

        return symbol_tables




            

            









