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

def calculate_x_list(target_idx, arg_idx, f, symbol_tables):
    # print(f"[calculate_x_list]: {f}")
    x = symbol_tables['x']
    # print(f"[calculate_x_list]: {f}")
    # print(f"[calculate_x_list]: {x.c}\n\t {x.delta}")
    # print(f"in cal: {x.c.shape}, {x.delta.shape}")
    # print(f"[in cal]: {x.c.detach().cpu().numpy().tolist()}\n\t {x.delta.detach().cpu().numpy().tolist()}") 
    input = x.select_from_index(1, arg_idx)
    # print(f"in cal: input: {input.c.shape}, {input.delta.shape}")
    res = f(input)
    # print(f, res.c.shape, x.c.shape, target_idx)
    x.c[:, target_idx] = res.c 
    x.delta[:, target_idx] = res.delta
    # print(f"in cal: x:{x.c.shape}, {x.delta.shape}")
    # exit(0)
    # print(f"[calculate_x_list]: res: {x.c.detach().cpu().numpy().tolist()}\n\t {x.delta.detach().cpu().numpy().tolist()}") 
    symbol_tables['x'] = x

    return symbol_tables
        

def calculate_branch(target_idx, test, symbol_tables):
    body_symbol_tables, orelse_symbol_tables = dict(), dict()
    x = symbol_tables['x']
    target = x.select_from_index(1, target_idx) # select the batch target from x

    # select the idx of left = target.left < test,  right = target.right >= test
    # select the trajectory accordingly
    # select the idx accordingly
    # split the other

    left = target.getLeft() <= test
    # if debug:
    #     print(f"left: {left}")  
    if True in left: # split to left
        left_idx = left.nonzero(as_tuple=True)[0].tolist()
        # x_left.c, x_left.delta = x.c[left.squeeze()], x.delta[left.squeeze()]
        # print(f"left: {left}")
        x_left = domain.Box(x.c[left.squeeze(1)], x.delta[left.squeeze(1)])
        # x_left =  domain.Box(x.c[left[0]], x.delta[left[0]])
        # print(f"x_left: {x_left.c},\n\t {x_left.delta}")
        left_target_c, left_target_delta = target.c[left].unsqueeze(1), target.delta[left].unsqueeze(1)
        # print(f"left target:", left_target_c, left_target_delta, test)

        # get the new c, delta
        new_left_target_c = ((left_target_c - left_target_delta) + torch.min((left_target_c + left_target_delta), test)) / 2.0
        # print(left_target_c + left_target_delta, test, torch.min((left_target_c + left_target_delta), test), (left_target_c - left_target_delta))
        # print((torch.min((left_target_c + left_target_delta), test) - (left_target_c - left_target_delta)))
        new_left_target_delta = (torch.min((left_target_c + left_target_delta), test) - (left_target_c - left_target_delta)) / 2.0
        # print(f"new left: c: {new_left_target_c}, delta: {new_left_target_delta}")
        x_left.c[:, target_idx:target_idx+1] = new_left_target_c
        x_left.delta[:, target_idx:target_idx+1] = new_left_target_delta
        # print(f"x_left after update: {x_left.c}, \n\t{x_left.delta}")
        body_symbol_tables['x'] = x_left
        body_symbol_tables['trajectory_list'] = [symbol_tables['trajectory_list'][i] for i in left_idx]
        body_symbol_tables['idx_list'] = [symbol_tables['idx_list'][i] for i in left_idx]
        # exit(0)
    
    right = target.getRight() > test
    # if debug:
    #     print(f"right:  {right}")
    if True in right: # split to right
        # exit(0)
        right_idx = right.nonzero(as_tuple=True)[0].tolist()
        x_right = domain.Box(x.c[right.squeeze(1)], x.delta[right.squeeze(1)])
        # print(f"x_right: {x_right.c}, {x_right.delta}")
        right_target_c, right_target_delta = target.c[right].unsqueeze(1), target.delta[right].unsqueeze(1)

        new_right_target_c = (torch.max((right_target_c - right_target_delta), test) + (right_target_c + right_target_delta)) / 2.0
        new_right_target_delta = ((right_target_c + right_target_delta) - torch.max((right_target_c - right_target_delta), test)) / 2.0
        x_right.c[:, target_idx:target_idx+1] = new_right_target_c
        x_right.delta[:, target_idx:target_idx+1] = new_right_target_delta

        # print(f"x_right after update: {x_right.c}, {x_right.delta}")

        orelse_symbol_tables['x'] = x_right
        orelse_symbol_tables['trajectory_list'] = [symbol_tables['trajectory_list'][i] for i in right_idx]
        orelse_symbol_tables['idx_list'] = [symbol_tables['idx_list'][i] for i in right_idx]
    
    # if len(body_symbol_tables) != 0 and len(orelse_symbol_tables) != 0:
    #     print(left_idx, right_idx)
    #     print(f"body: {body_symbol_tables['trajectory_list'][0][:3]}")
    #     print(f"orelse: {orelse_symbol_tables['trajectory_list'][0][40:43]}")
    #     exit(0)

    return body_symbol_tables, orelse_symbol_tables


def sound_join_trajectory_cost_mem(trajectory_1, trajectory_2):
    l1, l2 = len(trajectory_1), len(trajectory_2)
    # show_cuda_memory(f"ini join T")
    trajectory = list()
    for idx in range(min(l1, l2)):
        states_1, states_2 = trajectory_1[idx], trajectory_2[idx]
        state_list = list()
        for state_idx, v in enumerate(states_1):
            # show_cuda_memory(f"before sound join state")
            state_1, state_2 = states_1[state_idx], states_2[state_idx]
            # add 1024???? why
            # show_cuda_memory(f"before sound join")
            a = state_1.soundJoin(state_2)
            # show_cuda_memory(f"after sound join")
            state_list.append(a)
            del state_1
            del state_2
            
            # show_cuda_memory(f"after sound join before del idx")
            # torch.cuda.empty_cache()
            del states_1[state_idx]
            del states_2[state_idx]
            torch.cuda.empty_cache()
            # show_cuda_memory(f"after sound join after del idx")
            # exit(0)

            # show_cuda_memory(f"after sound join state")
        trajectory.append(state_list)
        # show_cuda_memory(f"mid join T (step {idx})")
        # del states_1
        # del states_2
        # del state_list
        # torch.cuda.empty_cache()
    # del trajectory
    # show_cuda_memory(f"mid join T")
    # exit(0)
    if l1 < l2:
        trajectory.extend(trajectory_2[l1:])
    elif l1 > l2:
        trajectory.extend(trajectory_1[l2:])
    
    # del trajectory_1
    # del trajectory_2
    # gc.collect()
    # show_cuda_memory(f"after join T")
    
    return trajectory


def sound_join_trajectory(trajectory_1, trajectory_2):
    l1, l2 = len(trajectory_1), len(trajectory_2)
    trajectory = list()
    # print(f"[sound_join_trajectory] start: {l1}, {l2}")
    K = min(l1, l2)
    for idx in range(K):
        states_1, states_2 = trajectory_1[0], trajectory_2[0]
        l_s = len(states_1)
        state_list = list()
        # print(l_s)
        # print(states_1)
        for state_idx in range(l_s):
            state_1, state_2 = states_1[0], states_2[0]
            a = state_1.soundJoin(state_2)
            state_list.append(a)
            # show_cuda_memory(f"after sound join")
        trajectory.append(state_list)
        # del trajectory_1[0]
        # del trajectory_2[0]
    
    if l1 < l2:
        trajectory.extend(trajectory_2[l2 - 1:])
    elif l1 > l2:
        trajectory.extend(trajectory_1[l1 - 1:])
    assert(len(trajectory) == max(l1, l2))
    # print(f"[sound_join_trajectory] end: {len(trajectory)}")

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
    # show_cuda_memory(f"[sound join] ini")
    # show_memory_snapshot()

    res_symbol_tables = dict()
    idx1, idx2 = 0, 0
    idx_list_1, idx_list_2 = symbol_tables_1['idx_list'], symbol_tables_2['idx_list']
    # print(f"in sound join: {idx_list_1}, {idx_list_2}")
    # print(f"test address: [1], [2]")
    # for state in symbol_tables_1["trajectory_list"][0]:
    #     print(state)
    # print(f"2")
    # for state in symbol_tables_2["trajectory_list"][0]:
    #     print(state)

    same = False
    while idx1 <= len(idx_list_1) - 1 or idx2 <= len(idx_list_2) - 1:
        if idx1 > len(idx_list_1) - 1 or (idx2 <= len(idx_list_2) - 1 and idx_list_1[idx1] > idx_list_2[idx2]):
            # print(f"ini sound join tra [table2]")
            # for key in symbol_tables_2:
            #     del symbol_tables_2[key]
            # exit(0)
            new_c = symbol_tables_2['x'].c[idx2:idx2+1].clone()
            new_delta = symbol_tables_2['x'].delta[idx2:idx2+1].clone()
            new_trajectory = symbol_tables_2['trajectory_list'][idx2]
            new_idx = symbol_tables_2['idx_list'][idx2]
            # show_cuda_memory(f"single idx1")
            # print(f"before sound join tra [table2]")
            # for key in symbol_tables_2:
            #     del symbol_tables_2[key]
            # exit(0)
            res_symbol_tables = update_joined_tables(res_symbol_tables, new_c, new_delta, new_trajectory, new_idx)
            # show_cuda_memory(f"single idx1 after join")
            idx2 += 1
        elif idx2 > len(idx_list_2) - 1 or (idx1 <= len(idx_list_1) - 1 and idx_list_1[idx1] < idx_list_2[idx2]):
            new_c = symbol_tables_1['x'].c[idx1:idx1+1].clone()
            new_delta = symbol_tables_1['x'].delta[idx1:idx1+1].clone()
            new_trajectory = symbol_tables_1['trajectory_list'][idx1]
            new_idx = symbol_tables_1['idx_list'][idx1]
            # show_cuda_memory(f"single idx2")
            # print(f"before sound join tra [table1]")
            # for key in symbol_tables_1:
            #     del symbol_tables_1[key]
            # exit(0)
            res_symbol_tables = update_joined_tables(res_symbol_tables, new_c, new_delta, new_trajectory, new_idx)
            # show_cuda_memory(f"single idx2 after join")
            idx1 += 1
        else: # idx_list_1[idx_1] == idx_list_2[idx_2], need to join
            assert(idx_list_1[idx1] == idx_list_2[idx2])
            new_left = torch.min(symbol_tables_1['x'].c[idx1:idx1+1] - symbol_tables_1['x'].delta[idx1:idx1+1], symbol_tables_2['x'].c[idx2:idx2+1] - symbol_tables_2['x'].delta[idx2:idx2+1])
            new_right = torch.max(symbol_tables_1['x'].c[idx1:idx1+1] + symbol_tables_1['x'].delta[idx1:idx1+1], symbol_tables_2['x'].c[idx2:idx2+1] + symbol_tables_2['x'].delta[idx2:idx2+1])
            new_c = (new_left + new_right) / 2.0
            new_delta = (new_right - new_left) / 2.0
            # show_cuda_memory(f"idx1 and idx2 before trajectory")
            # print(len(symbol_tables_1['trajectory_list'][idx1]), len(symbol_tables_2['trajectory_list'][idx2]))
            # print(f"before sound join tra")
            # for key in symbol_tables_1:
            #     del symbol_tables_1[key]
            # exit(0)
            # print(f"extract trajectory: {idx1}, {idx2}")
            # for state in symbol_tables_1["trajectory_list"][idx1]:
            #     print(state)
            # print(f"symbol_table2")
            # for state in symbol_tables_2["trajectory_list"][idx2]:
            #     print(state)
            new_trajectory = sound_join_trajectory(symbol_tables_1['trajectory_list'][idx1], symbol_tables_2['trajectory_list'][idx2])
            new_idx = idx_list_1[idx1]
            # show_cuda_memory(f"idx1 and idx2 after trajectory")
            res_symbol_tables = update_joined_tables(res_symbol_tables, new_c, new_delta, new_trajectory, new_idx)
            # show_cuda_memory(f"idx1 and idx2 after join tables")
            idx2 += 1
            idx1 += 1
            # del new_leftp
            # del new_right
            # del new_c
            # del new_delta
            same = True
            # print(sys.getsizeof(symbol_tables_1))
            # for key in symbol_tables_1:
            #     del symbol_tables_1[key]
            # exit(0)
    
    for key in list(symbol_tables_1.keys()):
        del symbol_tables_1[key]
    for key in list(symbol_tables_2.keys()):
        del symbol_tables_2[key]

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
        # show_cuda_memory(f"[assign: {self.f}]before cal x")
        res_symbol_tables = calculate_x_list(self.target_idx, self.arg_idx, self.f, symbol_tables)
        # show_cuda_memory(f"[assign]after cal x")
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
        # show_cuda_memory(f"[ifelse]before calculate branch")
        # print(f"[ifelse]before calculate branch, test: {test}")
        # print(f"all: {symbol_tables['x'].c}, \n\t{symbol_tables['x'].delta}")
        body_symbol_tables, orelse_symbol_tables = calculate_branch(self.target_idx, self.test, symbol_tables)
        # show_cuda_memory(f"[ifelse]after calculate branch")
        # print(f"{len(branch_list)}")
        # print(f"In IfElse, {len(body_list)}, {len(else_list)}")

        if len(body_symbol_tables) > 0:
            # show_cuda_memory(f"[ifelse]before body")
            # print(f"body: {body_symbol_tables['x'].c}, \n\t{body_symbol_tables['x'].delta}")
            body_symbol_tables = self.body(body_symbol_tables)
            # print(f"[ifelse]before sound join")
            # print(f"body: {body_symbol_tables['x'].c}, \n\t{body_symbol_tables['x'].delta}")
            
            # show_cuda_memory(f"[ifelse]after body")
        if len(orelse_symbol_tables) > 0:
            # show_cuda_memory(f"[ifelse]before orelse")
            # print(f"else: {orelse_symbol_tables['x'].c}, \n\t{orelse_symbol_tables['x'].delta}")
            orelse_symbol_tables = self.orelse(orelse_symbol_tables)
            # show_cuda_memory(f"[ifelse]end orelse")
            # print(f"[ifelse]before sound join")
            # print(f"else: {orelse_symbol_tables['x'].c}, \n\t{orelse_symbol_tables['x'].delta}")
        
        # show_cuda_memory(f"[ifelse]before sound join")
        # print(f"body: {body_symbol_tables['x'].c}, \n\t{body_symbol_tables['x'].delta}")
        # print(f"else: {orelse_symbol_tables['x'].c}, \n\t{orelse_symbol_tables['x'].delta}")
        res_symbol_tables = sound_join(body_symbol_tables, orelse_symbol_tables)
        # show_cuda_memory(f"[ifelse]after sound join")
        # print(f"length of res_symbol_tables: {len(res_symbol_tables)}")
        # print(f"[ifelse]after sound join")
        # print(f"all: {res_symbol_tables['x'].c}, \n\t{res_symbol_tables['x'].delta}")
        # if float(test) <= 1.21 and float(test) > 1.19:
        #     exit(0)

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
        # show_cuda_memory(f"ini while")
        while(len(symbol_tables) > 0):
            body_symbol_tables, orelse_symbol_tables = calculate_branch(self.target_idx, self.test, symbol_tables)

            # show_cuda_memory(f"[while]before sound join")
            res_symbol_tables = sound_join(res_symbol_tables, orelse_symbol_tables)
            # print(f"[after sound join] len res: {len(res_symbol_tables)}, len body: {len(body_symbol_tables)}, len orelse: {len(orelse_symbol_tables)}")
            # show_cuda_memory(f"[while {i}]after sound join")

            if len(body_symbol_tables) == 0:
                # print(f"[before return] len res: {len(res_symbol_tables)}, len body: {len(body_symbol_tables)}, len orelse: {len(orelse_symbol_tables)}")
                return res_symbol_tables
            
            symbol_tables = self.body(body_symbol_tables)

            i += 1
            if i > MAXIMUM_ITERATION:
                # print(f"Exceed maximum iterations: Have to END.")
                break
        # show_cuda_memory(f"[while {i}] before last sound join")
        # print(f"before first sound join, len body: {len(body_symbol_tables)}, len orelse: {len(orelse_symbol_tables)}, len res: {len(res_symbol_tables)}")
        res_symbol_tables = sound_join(res_symbol_tables, orelse_symbol_tables)
        # show_cuda_memory(f"[while {i}] after last sound join")
        # print(f"after first sound join, len body: {len(body_symbol_tables)}, len orelse: {len(orelse_symbol_tables)}, len res: {len(res_symbol_tables)}")
        res_symbol_tables = sound_join(res_symbol_tables, body_symbol_tables)
        # print(f"after second sound join, len body: {len(body_symbol_tables)}, len orelse: {len(orelse_symbol_tables)}, len res: {len(res_symbol_tables)}")
        return res_symbol_tables


class Trajectory(nn.Module):
    def __init__(self, target_idx):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, symbol_tables, cur_sample_size=0):
        x = symbol_tables['x']
        trajectory_list = symbol_tables['trajectory_list']
        # print(f"in update trajectory")
        
        B, D = x.c.shape
        for x_idx in range(B):
            cur_x_c, cur_x_delta = x.c[x_idx], x.delta[x_idx]
            input_interval_list = list()
            for idx in self.target_idx:
                input = domain.Box(cur_x_c[idx], cur_x_delta[idx])
                input_interval = input.getInterval()

                assert input_interval.left.data.item() <= input_interval.right.data.item()
                input_interval_list.append(input_interval)
            trajectory_list[x_idx].append(input_interval_list)
        
        symbol_tables['trajectory_list'] = trajectory_list

        return symbol_tables




            

            









