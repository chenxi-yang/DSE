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

def calculate_abstract_state(target_idx, arg_idx, f, abstract_state):
    # assign_time = time.time()
    for idx, symbol_table in enumerate(abstract_state):
        x = symbol_table['x']
        input = x.select_from_index(0, arg_idx) # torch.index_select(x, 0, arg_idx)
        # print(f"f: {f}")
        res, p = f(input)
        # print(f"calculate_x_list --  target_idx: {target_idx}, res: {res.c}, {res.delta}")
        x.set_from_index(target_idx, res) # x[target_idx[0]] = res
        
        symbol_table['x'] = x
        #! probability of each component does not change
        # symbol_table['probability'] = symbol_table['probability'].mul(p)
        abstract_state[idx] = symbol_table
    # print(f"-- assign -- calculate_x_list: {time.time() - assign_time}")
    return abstract_state


def calculate_abstract_state_list(target_idx, arg_idx, f, abstract_state_list):
    res_list = list()
    for abstract_state in abstract_state_list:
        res_abstract_state = calculate_abstract_state(target_idx, arg_idx, f, abstract_state)
        res_list.append(res_abstract_state)
    return res_list


def pre_build_symbol_table(symbol_table):
    # clone trajectory
    res_symbol_table = dict()
    res_symbol_table['trajectory'] = list()
    for state in symbol_table['trajectory']:
        res_symbol_table['trajectory'].append(state)

    return res_symbol_table


def pre_allocate(symbol_table):
    return symbol_table['probability']


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


def split_branch_symbol_table(target_idx, test, symbol_table):
    body_symbol_table, orelse_symbol_table = dict(), dict()
    branch_time = time.time()

    x = symbol_table['x']
    target = x.select_from_index(0, target_idx)
    # res_symbol_table = pre_build_symbol_table(symbol_table)

    if target.getRight().data.item() <= test.data.item():
        res = x.clone()
        branch = 'body'
        body_symbol_table = pre_build_symbol_table(symbol_table)
        probability = pre_allocate(symbol_table)
        body_symbol_table = update_res_in_branch(body_symbol_table, res, probability, branch)

    elif target.getLeft().data.item() > test.data.item():
        res = x.clone()
        branch = 'orelse'
        orelse_symbol_table = pre_build_symbol_table(symbol_table)
        probability = pre_allocate(symbol_table)
        orelse_symbol_table = update_res_in_branch(orelse_symbol_table, res, probability, branch)

    else:
        res = x.clone()
        branch = 'body'
        c = (target.getLeft() + test) / 2.0
        delta = (test - target.getLeft()) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta)) # res[target_idx] = Box(c, delta)
        body_symbol_table = pre_build_symbol_table(symbol_table)
        # in DSE, each component's probability upper bound is kept
        probability = pre_allocate(body_symbol_table)
        body_symbol_table = update_res_in_branch(body_symbol_table, res, probability, branch)

        res = x.clone()
        branch = 'orelse'
        c = (target.getRight() + test) / 2.0
        delta = (target.getRight() - test) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta))
        orelse_symbol_table = pre_build_symbol_table(symbol_table)
        # in DSE, each component's probability upper bound is kept
        probability = pre_allocate(orelse_symbol_table)
        orelse_symbol_table = update_res_in_branch(orelse_symbol_table, res, probability, branch)

    # print(f"branch time: {time.time() - branch_time}")
    return body_symbol_table, orelse_symbol_table


def split_branch_abstract_state(target_idx, test, abstract_state):
    body_abstract_state, orelse_abstract_state = list(), list()
    for symbol_table in abstract_state:
        body_symbol_table, orelse_symbol_table = split_branch_symbol_table(target_idx, test, symbol_table)
        if len(body_symbol_table) > 0:
            body_abstract_state.append(body_symbol_table)
        if len(orelse_symbol_table) > 0:
            orelse_abstract_state.append(orelse_symbol_table)
    return body_abstract_state, orelse_abstract_state
            

def calculate_branch_list(target_idx, test, abstract_state_list):
    res_abstract_state_list = list()
    assert(len(abstract_state_list) == 0)
    for abstract_state in abstract_state_list: 
        body_abstract_state, orelse_abstract_state = split_branch_abstract_state(target_idx, test, abstract_state)
        if len(body_abstract_state) > 0:
            res_abstract_state_list.append(body_abstract_state)
        if len(orelse_abstract_state) > 0:
            res_abstract_state_list.append(orelse_abstract_state)
    return res_abstract_state_list


def calculate_abstract_state_weight(abstract_state):
    # abstract state's probability equalling to the sum of the upper bound probability of all components
    return sum([symbol_table['probability'].data.item() for symbol_table in abstract_state])


def sample(abstract_state_list):
    # only sample one abstract state
    assert(len(abstract_state_list) <= 2)
    # sample_time = time.time()
    res_symbol_table_list = random.choices(abstract_state_list, weights=[calculate_abstract_state_weight(abstract_state) for abstract_state in abstract_state_list], k=1)
    # print(f"sampling time: {time.time() - sample_time}") # , length before: {length_before}, length after: {len(res_symbol_table_list)}")

    return res_symbol_table_list


class Skip(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, abstract_state_list, cur_sample_size=0):
        return abstract_state_list


class Assign(nn.Module):
    def __init__(self, target_idx, arg_idx: list(), f):
        super().__init__()
        self.f = f
        self.target_idx = torch.tensor(target_idx)
        self.arg_idx = torch.tensor(arg_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
            self.arg_idx = self.arg_idx.cuda()
    
    def forward(self, abstract_state_list, cur_sample_size=0):
        # print(f"Assign Before: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        res_list = calculate_abstract_state_list(self.target_idx, self.arg_idx, self.f, abstract_state_list)
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
    
    def forward(self, abstract_state_list):
        # print(f"Ifelse: target_idx: {self.target_idx}")
        # print(f"############one ifelse ##################")

        test = self.f_test(self.test)
        res_list = calculate_branch_list(self.target_idx, test, x_list)

        res_list = sample(res_list) # sample before executing

        assert(len(res_list) == 0)
        # the first component in the first abstract state represents the res_list branch
        if res_list[0][0]['branch'] == 'body':
            res_list = self.body(res_list)
        else:
            res_list = self.orelse(res_list)

        return res_list


class While(nn.Module):
    def __init__(self, target_idx, test, body):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.body = body
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, abstract_state_list):

        while(len(abstract_state_list) > 0):
            pre_abstract_state_list = calculate_branch_list(self.target_idx, self.test, abstract_state_list)
            res_abstract_state_list = sample(pre_abstract_state_list)

            assert(len(res_abstract_state_list) == 0)

            if res_abstract_state_list[0][0]['branch'] == 'body': # if the abstract state in  res_abstract_state_list falls into 'body'
                abstract_state_list = self.body(res_abstract_state_list)
            else:
                return res_abstract_state_list

        return res_abstract_state_list


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


def update_abstract_state_trajectory(abstract_state, target_idx):
    for idx, symbol_table in enumerate(abstract_state):
        symbol_table = update_trajectory(symbol_table, target_idx)
        abstract_state[idx] = symbol_table
    return abstract_state


class Trajectory(nn.Module):
    # TODO: update, add state in trajectory list
    def __init__(self, target_idx):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, abstract_state_list, cur_sample_size=0):
        for idx, abstract_state in enumerate(abstract_state_list):
            abstract_state = update_abstract_state_trajectory(abstract_state, self.target_idx)
            abstract_state_list[idx] = abstract_state
        return abstract_state_list




            

            









