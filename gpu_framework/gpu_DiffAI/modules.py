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

def calculate_x_list(target_idx, arg_idx, f, symbol_table_list):
    # assign_time = time.time()
    for idx, symbol_table in enumerate(symbol_table_list):
        x = symbol_table['x']
        input = x.select_from_index(0, arg_idx) # torch.index_select(x, 0, arg_idx)
        # print(f"f: {f}")
        res = f(input)
        # print(f"calculate_x_list --  target_idx: {target_idx}, res: {res.c}, {res.delta}")
        x.set_from_index(target_idx, res) # x[target_idx[0]] = res
        
        symbol_table['x'] = x
        # symbol_table['probability'] = symbol_table['probability'].mul(p)
        symbol_table_list[idx] = symbol_table
    # print(f"-- assign -- calculate_x_list: {time.time() - assign_time}")
    return symbol_table_list


def pre_build_symbol_table(symbol_table):
    # clone safe_range and x_memo_list
    res_symbol_table = dict()
    res_symbol_table['trajectory'] = list()
    for state in symbol_table['trajectory']:
        res_symbol_table['trajectory'].append(state)

    return res_symbol_table


def pre_allocate(symbol_table):
    return symbol_table['probability']


def split_point_cloud(symbol_table, res, target_idx):
    # split the point cloud, count and calculate the probability
    counter = 0.0
    old_point_cloud = symbol_table['point_cloud']
    point_cloud = list()
    for point in old_point_cloud:
        # TODO: brute-forcely check the point, a smarter way is to check based on the target_idx
        if res.check_in(point):
            counter += 1
            point_cloud.append(point)
    
    if counter > 0:
        probability = symbol_table['probability'].mul(var(counter).div(symbol_table['counter']))
    else:
        probability = SMALL_PROBABILITY
    counter = var(counter)

    return probability, counter, point_cloud


def split_volume(symbol_table, target, delta):
    # 
    target_volume = target.getRight() - target.getLeft()
    new_volume = delta.mul(var(2.0))
    probability = symbol_table['probability'].mul(new_volume.div(target_volume))

    return probability


def update_res_in_branch(res_symbol_table, res, probability, branch):
    res_symbol_table['x'] = res
    res_symbol_table['probability'] = probability
    res_symbol_table['branch'] = branch

    return res_symbol_table


def calculate_branch(target_idx, test, symbol_table):
    res_symbol_table_list = list()
    branch_time = time.time()
    # print(f"calculate branch -- target_idx: {target_idx}")
    # print(f"x: {x.c, x.delta}")

    x = symbol_table['x']
    target = x.select_from_index(0, target_idx)
    # res_symbol_table = pre_build_symbol_table(symbol_table)

    # target = x[target_idx]
    # print(f"target right: {target.getRight()}")
    # print(f"test: {test}")
    # pre allocate
    # probability = pre_allocate(symbol_table)

    if target.getRight().data.item() <= test.data.item():
        res = x.clone()
        branch = 'body'
        res_symbol_table = pre_build_symbol_table(symbol_table)
        probability = pre_allocate(symbol_table)
        res_symbol_table = update_res_in_branch(res_symbol_table, res, probability, branch)
        res_symbol_table_list.append(res_symbol_table)
    elif target.getLeft().data.item() > test.data.item():
        res = x.clone()
        branch = 'orelse'
        res_symbol_table = pre_build_symbol_table(symbol_table)
        probability = pre_allocate(symbol_table)
        res_symbol_table = update_res_in_branch(res_symbol_table, res, probability, branch)
        res_symbol_table_list.append(res_symbol_table)
    else:
        res = x.clone()
        branch = 'body'
        c = (target.getLeft() + test) / 2.0
        delta = (test - target.getLeft()) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta)) # res[target_idx] = Box(c, delta)
        res_symbol_table_body = pre_build_symbol_table(symbol_table)
        # This is DiffAI, so probability is not needed any more (for place holder)
        probability = pre_allocate(symbol_table)
        res_symbol_table_body = update_res_in_branch(res_symbol_table_body, res, probability, branch)
        res_symbol_table_list.append(res_symbol_table_body)

        res = x.clone()
        branch = 'orelse'
        c = (target.getRight() + test) / 2.0
        delta = (target.getRight() - test) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta))
        res_symbol_table_orelse = pre_build_symbol_table(symbol_table)

        probability = pre_allocate(symbol_table)
        res_symbol_table_orelse = update_res_in_branch(res_symbol_table_orelse, res, probability, branch)
        res_symbol_table_list.append(res_symbol_table_orelse)

    # print(f"branch time: {time.time() - branch_time}")
    return res_symbol_table_list
            

def calculate_branch_list(target_idx, test, symbol_table_list):
    res_list = list()
    for symbol_table in symbol_table_list: # for each element, split it. # c, delta
        # print(symbol_table)
        res_symbol_table = calculate_branch(target_idx, test, symbol_table)
        # if res_symbol_table['x'] is None:
        #     continue
        res_list.extend(res_symbol_table)
    return res_list


def sound_join_symbol_table(symbol_table_1, symbol_table_2):
    # TODO: trajectory join, one by one, every state do sound join one by one
    # assert(len(symbol_table_1) == 0 and len(symbol_table_2) == 0)
    # print(f"In Sound Join Symbol Table")
    if len(symbol_table_1) == 0:
        return symbol_table_2
    if len(symbol_table_2) == 0:
        return symbol_table_1
    trajectory_1, trajectory_2 = symbol_table_1['trajectory'], symbol_table_2['trajectory']
    res_trajectory = trajectory_2 if len(trajectory_1) < len(trajectory_2) else trajectory_1

    symbol_table = {
        'x': symbol_table_1['x'].sound_join(symbol_table_2['x']),
        'probability': torch.max(symbol_table_1['probability'], symbol_table_2['probability']),
        'trajectory': [state for state in res_trajectory],
        'branch': '',
    }
    # print(f"Out Sound Join Symbol Table: {symbol_table['trajectory']}")

    return symbol_table


def sound_join(l1, l2):
    # join all symbol_table, only one symbol_table left
    # when joining trajectory, select the trajectory with longer length, TODO in the future
    # print(f"In Sound Join")
    res_list = list()
    res_symbol_table = dict()
    # print(f"{len(l1)}, {len(l2)}")
    for symbol_table in l1:
        res_symbol_table = sound_join_symbol_table(res_symbol_table, symbol_table)
    for symbol_table in l2:
        res_symbol_table = sound_join_symbol_table(res_symbol_table, symbol_table)
    
    if len(res_symbol_table) > 1: # res_symbol_table not None
        # print(res_symbol_table['trajectory'])
        res_list.append(res_symbol_table)
        
    # print(f"Out Sound Join")

    return res_list


class Skip(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_list, cur_sample_size=0):
        return x_list


class Assign(nn.Module):
    def __init__(self, target_idx, arg_idx: list(), f):
        super().__init__()
        self.f = f
        self.target_idx = torch.tensor(target_idx)
        self.arg_idx = torch.tensor(arg_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
            self.arg_idx = self.arg_idx.cuda()
    
    def forward(self, x_list, cur_sample_size=0):
        # print(f"Assign Before: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        res_list = calculate_x_list(self.target_idx, self.arg_idx, self.f, x_list)
        # print(f"Assign After: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        return res_list


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
    
    def forward(self, x_list):
        test = self.f_test(self.test)

        branch_list = calculate_branch_list(self.target_idx, test, x_list)
        # print(f"{len(branch_list)}")
        # print(f"{[symbol_table['branch'] for symbol_table in branch_list]}")

        body_list, else_list = list(), list()
        for symbol_table in branch_list:
            if symbol_table['branch'] == 'body':
                body_list.append(symbol_table)
            else:
                else_list.append(symbol_table)
        
        # print(f"In IfElse, {len(body_list)}, {len(else_list)}")
        
        if len(body_list) > 0:
            body_list = self.body(body_list)
        if len(else_list) > 0:
            else_list = self.orelse(else_list)

        res_list = sound_join(body_list, else_list)

        return res_list


class While(nn.Module):
    def __init__(self, target_idx, test, body):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.body = body
        if torch.cuda.is_available():
            # print(f"CHECK: cuda")
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, symbol_table_list):
        '''
        super set of E_{i-th step} and [\neg condition]
        '''
        # print(f"##############In while DiffAI#########")
        i = 0
        res_list = list()
        while(len(symbol_table_list) > 0):
            # counter += 1
            branch_list = calculate_branch_list(self.target_idx, self.test, symbol_table_list)
            body_list, else_list = list(), list()
            for symbol_table in branch_list:
                if symbol_table['branch'] == 'body':
                    body_list.append(symbol_table)
                else:
                    else_list.append(symbol_table)

            res_list = sound_join(res_list, else_list)

            if len(body_list) == 0:
                # print(f"---In While Out, {len(res_list)}, {res_list[0]['trajectory']}")
                return res_list
            
            symbol_table_list = self.body(body_list)

            i += 1
            if i > 1000:
                print(f"Exceed maximum iterations: Have to END.")
                break

        return res_list


def update_trajectory(symbol_table, target_idx):
    input = symbol_table['x'].select_from_index(0, target_idx)
    input_interval = input.getInterval()
    # print(f"input: {input.c, input.delta}")
    # print(f"input_interval: {input_interval.left.data.item(), input_interval.right.data.item()}")
    assert input_interval.left.data.item() <= input_interval.right.data.item()

    # print(f"In update trajectory")
    symbol_table['trajectory'].append(input_interval)
    # print(f"Finish update trajectory")

    return symbol_table


class Trajectory(nn.Module):
    # TODO: update, add state in trajectory list
    def __init__(self, target_idx):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, symbol_table_list, cur_sample_size=0):
        for idx, symbol_table in enumerate(symbol_table_list):
            symbol_table = update_trajectory(symbol_table, self.target_idx)
            symbol_table_list[idx] = symbol_table
        return symbol_table_list




            

            









