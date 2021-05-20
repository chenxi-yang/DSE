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
        # print(f"x: {x.shape}, weight: {self.weight.shape}, bias: {self.bias.shape}")
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

def calculate_abstract_state(target_idx, arg_idx, f, abstract_state):
    # assign_time = time.time()
    for idx, symbol_table in enumerate(abstract_state):
        x = symbol_table['x']
        input = x.select_from_index(0, arg_idx) # torch.index_select(x, 0, arg_idx)
        # print(f"f: {f}")
        res = f(input)
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
        # state_list = list()
        # for state_idx in len(state):
        #     state_list.append(state[state_idx].clone())
        # res_symbol_table['trajectory'].append(state_list)
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
        probability = pre_allocate(symbol_table)
        body_symbol_table = update_res_in_branch(body_symbol_table, res, probability, branch)

        res = x.clone()
        branch = 'orelse'
        c = (target.getRight() + test) / 2.0
        delta = (target.getRight() - test) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta))
        orelse_symbol_table = pre_build_symbol_table(symbol_table)
        # in DSE, each component's probability upper bound is kept
        probability = pre_allocate(symbol_table)
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
    assert(len(abstract_state_list) == 1)
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


def extract_maximum_p(weights_arg_idx, abstract_state):
    p_list = None
    for symbol_table in abstract_state:
        x = symbol_table['x']
        p = x.select_from_index(0, weights_arg_idx)
        if p_list is None:
            p_list = p.c + p.delta
        else:
            p_list = torch.max(p.c + p.delta, p_list)

    return p_list


def extract_new_symbol_table(target_idx, v, p, symbol_table):
    # v is the sampled value
    # p is the sampled probability
    # update trajectory
    res_symbol_table = pre_build_symbol_table(symbol_table)
    x = symbol_table['x']
    res = x.clone()
    res.set_from_index(target_idx, domain.Box(var_list([v]), var_list([0.0])))
    probability = symbol_table['probability'].mul(p)
    branch = symbol_table['branch']

    res_symbol_table = update_res_in_branch(res_symbol_table, res, probability, branch)
    return res_symbol_table


def split_abstract_state(
    target_idx,
    sample_population,
    weights_arg_idx,
    abstract_state,
    ):
    # split abstract state according to len(sample_population)
    p_list = extract_maximum_p(weights_arg_idx, abstract_state)
    res_abstract_state = list()
    for idx, p in enumerate(p_list):
        for symbol_table in abstract_state:
            new_split_symbol_table = extract_new_symbol_table(
                target_idx, 
                sample_population[idx],
                p,
                symbol_table
                )
            res_abstract_state.append(new_split_symbol_table)

    return res_abstract_state


class Skip(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, abstract_state_list, cur_sample_size=0):
        return abstract_state_list
    

class Sampler(nn.Module):
    def __init__(self, target_idx, sample_population, weights_arg_idx):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.sample_population = sample_population
        self.weights_arg_idx = torch.tensor(weights_arg_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
            self.weights_arg_idx = self.weights_arg_idx.cuda()
    
    def forward(self, abstract_state_list):
        # extend each abstract state for one dimension according to target_idx
        # # component = # components * len(sample_population)
        res_abstract_state_list = list()
        for abstract_state in abstract_state_list:
            res_abstract_state = split_abstract_state(self.target_idx, self.sample_population, self.weights_arg_idx, abstract_state)
            res_abstract_state_list.append(res_abstract_state)
        return res_abstract_state_list
        

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
        # print(f"Assign Before {self.f}: {[(res['x'].c, res['x'].delta) for res in abstract_state_list[0]]}")
        res_list = calculate_abstract_state_list(self.target_idx, self.arg_idx, self.f, abstract_state_list)
        # print(f"Assign After {self.f}: {[(res['x'].c, res['x'].delta) for res in res_list[0]]}")
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
        # print(f"if else test: {test}")
        res_list = calculate_branch_list(self.target_idx, test, abstract_state_list)
        # print(f"If else:")
        # for abstract_state in res_list:
        #     for symbol_table in abstract_state:
        #         print(f"symbol_table: {symbol_table['x'].c}, {symbol_table['x'].delta}, {symbol_table['branch']}")

        res_list = sample(res_list) # sample before executing
        # print(f"If else after sampling:")
        # for abstract_state in res_list:
        #     for symbol_table in abstract_state:
        #         print(f"symbol_table: {symbol_table['x'].c}, {symbol_table['x'].delta}, {symbol_table['branch']}")

        assert(len(res_list) == 1)
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
        i = 0
        while(len(abstract_state_list) > 0):
            pre_abstract_state_list = calculate_branch_list(self.target_idx, self.test, abstract_state_list)
            # print(f"pre_abstract_state_list:")
            # for abstract_state in pre_abstract_state_list:
            #     for symbol_table in abstract_state:
            #         print(f"symbol_table: {symbol_table['x'].c}, {symbol_table['x'].delta}, {symbol_table['branch']}")

            res_abstract_state_list = sample(pre_abstract_state_list)
            # for abstract_state in res_abstract_state_list:
            # print(f"after sample:")
            # for abstract_state in res_abstract_state_list:
            #     for symbol_table in abstract_state:
            #         print(f"symbol_table: {symbol_table['x'].c}, {symbol_table['x'].delta}, {symbol_table['branch']}")

            assert(len(res_abstract_state_list) == 1)

            if res_abstract_state_list[0][0]['branch'] == 'body': # if the abstract state in  res_abstract_state_list falls into 'body'
                abstract_state_list = self.body(res_abstract_state_list)
            else:
                # print(f"Return!")
                # for abstract_state in res_abstract_state_list:
                #     for symbol_table in abstract_state:
                #         print(f"symbol_table: {symbol_table['x'].c}, {symbol_table['x'].delta}, {symbol_table['branch']}")
                return res_abstract_state_list
            
            i += 1
            if i > MAXIMUM_ITERATION:
                # res_abstract_state = res_abstract_state_list[0]
                # for symbol_table in res_abstract_state:
                #     print(symbol_table['x'].c, symbol_table['x'].delta)
                # print(f"Exceed maximum iterations: Have to END.")
                break
        
        # print(f"Normal Return!")
        # for abstract_state in res_abstract_state_list:
        #     for symbol_table in abstract_state:
        #         print(f"symbol_table: {symbol_table['x'].c}, {symbol_table['x'].delta}, {symbol_table['branch']}")
            
        return res_abstract_state_list


def update_trajectory(symbol_table, target_idx):
    # trajectory: list of states
    # states: list of intervals
    input_interval_list = list()
    # print(f"all symbol_table: {symbol_table['x'].c, symbol_table['x'].delta}")
    for idx in target_idx:
        input = symbol_table['x'].select_from_index(0, idx)
        input_interval = input.getInterval()
        # print(f"idx:{idx}, input: {input.c, input.delta}")
        # print(f"input_interval: {input_interval.left.data.item(), input_interval.right.data.item()}")
        assert float(input_interval.left) <= float(input_interval.right)
        # if float(input_interval.left) == float(input_interval.right):
        #     print('find point:', float(input_interval.left), float(input_interval.right))
        #     exit(0)
        input_interval_list.append(input_interval)
    # print(f"In update trajectory")
    symbol_table['trajectory'].append(input_interval_list)
    # exit(0)
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




            

            









