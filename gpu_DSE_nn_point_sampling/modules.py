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

def update_point_cloud(target_idx, arg_idx, f, symbol_table):
    # point_cloud_time = time.time()
    old_point_cloud = symbol_table['point_cloud']
    new_point_cloud = list()

    for point in old_point_cloud:
        input = point.select_from_index(0, arg_idx)
        res = f(input)
        point.set_from_index(target_idx, res)
        new_point_cloud.append(point)
    
    # print(f"-- update {len(old_point_cloud)} size point cloud --: {time.time() - point_cloud_time}")
    
    return new_point_cloud


def calculate_x_list(target_idx, arg_idx, f, symbol_table_list):
    # assign_time = time.time()
    for idx, symbol_table in enumerate(symbol_table_list):
        x = symbol_table['x']
        input = x.select_from_index(0, arg_idx) # torch.index_select(x, 0, arg_idx)
        # print(f"f: {f}")
        res, p = f(input)
        # print(f"calculate_x_list --  target_idx: {target_idx}, res: {res.c}, {res.delta}")
        x.set_from_index(target_idx, res) # x[target_idx[0]] = res
        
        symbol_table['x'] = x
        # TODO: use point cloud
        symbol_table['point_cloud'] = update_point_cloud(target_idx, arg_idx, f, symbol_table)
        symbol_table['probability'] = symbol_table['probability'].mul(p)
        symbol_table_list[idx] = symbol_table
    # print(f"-- assign -- calculate_x_list: {time.time() - assign_time}")
    return symbol_table_list


def pre_build_symbol_table(symbol_table):
    # clone safe_range and x_memo_list
    res_symbol_table = dict()
    res_symbol_table['x_memo_list'] = list()
    for x_memo in symbol_table['x_memo_list']:
        res_symbol_table['x_memo_list'].append(x_memo.clone())
    res_symbol_table['safe_range'] = symbol_table['safe_range'].clone()

    return res_symbol_table


def pre_allocate(symbol_table):
    return symbol_table['probability'], symbol_table['counter'], symbol_table['point_cloud']


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


def update_res_in_branch(res_symbol_table, res, probability, counter, point_cloud, branch):
    res_symbol_table['x'] = res
    res_symbol_table['probability'] = probability
    res_symbol_table['explore_probability'] = probability
    res_symbol_table['counter'] = counter
    res_symbol_table['point_cloud'] = point_cloud
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
    probability, counter, point_cloud = pre_allocate(symbol_table)

    if target.getRight().data.item() <= test.data.item():
        res = x.clone()
        branch = 'body'
        res_symbol_table = pre_build_symbol_table(symbol_table)
        probability, counter, point_cloud = pre_allocate(symbol_table)
        res_symbol_table = update_res_in_branch(res_symbol_table, res, probability, counter, point_cloud, branch)
        res_symbol_table_list.append(res_symbol_table)
    elif target.getLeft().data.item() > test.data.item():
        res = x.clone()
        branch = 'orelse'
        res_symbol_table = pre_build_symbol_table(symbol_table)
        probability, counter, point_cloud = pre_allocate(symbol_table)
        res_symbol_table = update_res_in_branch(res_symbol_table, res, probability, counter, point_cloud, branch)
        res_symbol_table_list.append(res_symbol_table)
    else:
        res = x.clone()
        branch = 'body'
        c = (target.getLeft() + test) / 2.0
        delta = (test - target.getLeft()) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta)) # res[target_idx] = Box(c, delta)
        res_symbol_table_body = pre_build_symbol_table(symbol_table)
        probability, counter, point_cloud = split_point_cloud(symbol_table, res, target_idx)
        res_symbol_table_body = update_res_in_branch(res_symbol_table_body, res, probability, counter, point_cloud, branch)
        res_symbol_table_list.append(res_symbol_table_body)

        res = x.clone()
        branch = 'orelse'
        c = (target.getRight() + test) / 2.0
        delta = (target.getRight() - test) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta))
        res_symbol_table_orelse = pre_build_symbol_table(symbol_table)
        probability, counter, point_cloud = split_point_cloud(symbol_table, res, target_idx)
        res_symbol_table_orelse = update_res_in_branch(res_symbol_table_orelse, res, probability, counter, point_cloud, branch)
        res_symbol_table_list.append(res_symbol_table_orelse)

    # if b == 'body':
    #     branch = 'body'
    #     if target.getRight().data.item() <= test.data.item():
    #         res = x.clone()
    #     elif target.getLeft().data.item() > test.data.item():
    #         res = None
    #     else:
    #         res = x.clone()
    #         c = (target.getLeft() + test) / 2.0
    #         delta = (test - target.getLeft()) / 2.0
    #         res.set_from_index(target_idx, domain.Box(c, delta)) # res[target_idx] = Box(c, delta)
    #         probability, counter, point_cloud = split_point_cloud(symbol_table, res, target_idx)
    # else:
    #     branch = 'orelse'
    #     if target.getRight().data.item() <= test.data.item():
    #         res = None
    #     elif target.getLeft().data.item() > test.data.item():
    #         res = x.clone()
    #     else:
    #         res = x.clone()
    #         c = (target.getRight() + test) / 2.0
    #         delta = (target.getRight() - test) / 2.0
    #         res.set_from_index(target_idx, domain.Box(c, delta))
    #         probability, counter, point_cloud = split_point_cloud(symbol_table, res, target_idx)

    # res_symbol_table = update_res_in_branch(res_symbol_table, res, probability, counter, point_cloud, branch)

    # print(f"branch time: {time.time() - branch_time}")
    return res_symbol_table_list
            

def calculate_branch_list(target_idx, test, symbol_table_list):
    res_list = list()
    for symbol_table in symbol_table_list: # for each element, split it. # c, delta
        res_symbol_table = calculate_branch(target_idx, test, symbol_table)
        # if res_symbol_table['x'] is None:
        #     continue
        res_list.extend(res_symbol_table)
    return res_list


def adapt_sampling_distribution(res_symbol_table_list):
    # print(f"adapt res symbol_table: {len(res_symbol_table_list)}")
    adapt_time = time.time()
    if SAMPLE_METHOD == 3: # directly pass the probability
        # length = len(res_symbol_table_list)
        # tmp_explore_probability = res_symbol_table_list[0]['explore_probability']
        
        # for idx, res_symbol_table in enumerate(res_symbol_table_list):
        #     re_idx = (idx - c) % length
        #     if re_idx == 0:
        #         res_symbol_table_list[idx]['explore_probability'] = tmp_explore_probability
        #     else:
        #         res_symbol_table_list[idx]['explore_probability'] = res_symbol_table_list[re_idx]['explore_probability']
        pass
    
    if SAMPLE_METHOD == 4:
        # adaptive translation
        length = len(res_symbol_table_list)
        score_list = list()
        for res_symbol_table in res_symbol_table_list:
            x = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
            # TODO: should adapt to different domains
            x.left = torch.min(res_symbol_table['safe_range'].left, res_symbol_table['safe_range'].left)
            x.right = torch.max(res_symbol_table['safe_range'].right, res_symbol_table['safe_range'].right)
            score_list.append(get_score_gradient(res_symbol_table['x_memo_list'], x, target))

        score_idx_list = [x for x, y in sorted(enumerate(score_list), key = lambda x:x[1].data.item(), reverse=True)]
        probability_idx_list = [x for x, y in sorted(enumerate(res_symbol_table_list), key = lambda x:x[1]['explore_probability'].data.item(), reverse=True)]
        for idx, score_idx in enumerate(score_idx_list):
            # print(score_list[score_idx], res_symbol_table_list[probability_idx_list[idx]]['explore_probability'])
            #! change【change】
            # new weight p(x)/pi(x)
            res_symbol_table_list[score_idx]['explore_probability'] = res_symbol_table_list[probability_idx_list[idx]]['explore_probability']
            # res_symbol_table_list[score_idx]['explore_probability'] = res_symbol_table_list[score_idx]['probability'].div(res_symbol_table_list[probability_idx_list[idx]]['explore_probability'])
    # if len(res_symbol_table) > constants.SAMPLE_SIZE: 
    #     print(f"adapt time: {time.time() - adapt_time}")
    return res_symbol_table_list


def sample(symbol_table_list):
    sample_time = time.time()
    if SAMPLE_METHOD == 3:
        res_symbol_table_list = random.choices(symbol_table_list, weights=[symbol_table['probability'].data.item() for symbol_table in symbol_table_list], k=1)

    if SAMPLE_METHOD == 4:
        length_before = len(symbol_table_list)

        # just for pofiling
        # print(f"sampling time: {time.time() - sample_time}, length before: {length_before}")

        shuffle(symbol_table_list)
        res_symbol_table_list = list()

        symbol_table_idx = 0
        max_explore_probability = N_INFINITY
        for symbol_table in symbol_table_list:
            max_explore_probability = torch.max(symbol_table['explore_probability'], max_explore_probability)

        for idx, symbol_table in enumerate(symbol_table_list):
            symbol_table_list[idx]['explore_probability'] = symbol_table_list[idx]['explore_probability'].div(max_explore_probability)
        
        res_symbol_table_list = random.choices(symbol_table_list, weights=[symbol_table['explore_probability'].data.item() for symbol_table in symbol_table_list], k=1)
        
        for idx, symbol_table in enumerate(res_symbol_table_list):
            res_symbol_table_list[idx]['explore_probability'] = res_symbol_table_list[idx]['explore_probability'].mul(max_explore_probability)
        
    # print(f"sampling time: {time.time() - sample_time}") # , length before: {length_before}, length after: {len(res_symbol_table_list)}")

    return res_symbol_table_list


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
        # print(f"Ifelse: target_idx: {self.target_idx}")
        # print(f"############one ifelse ##################")
        res_list = list()
        test = self.f_test(self.test)
        # print(f"target_idx, {self.target_idx}")
        # print(f"test: {test}")
        # for x in x_list:
        #     print(f"x: {x['x'].c, x['x'].delta}")

        # if_branch_time = time.time()
        res_list = calculate_branch_list(self.target_idx, test, x_list)
        # print(f"-- ifelse -- branch: {time.time() - if_branch_time}")
        # orelse_list = calculate_branch_list(self.target_idx, test, x_list)
        # print(f"Length, body: {len(body_list)}, orelse: {len(orelse_list)}")

        # res_list.extend(body_list)
        # res_list.extend(orelse_list)
        
        # if_sample_time = time.time()
        res_list = adapt_sampling_distribution(res_list)
        res_list = sample(res_list)
        # print(f"-- ifelse -- sample: {time.time() - if_sample_time}")


        if res_list[0]['branch'] == 'body':
            # if_body_time = time.time()
            res_list = self.body(res_list)
            # print(f"-- ifelse -- body: {time.time() - if_body_time}")
        else:
            # if_orelse_time = time.time()
            res_list = self.orelse(res_list)
            # print(f"-- ifelse -- orelse: {time.time() - if_orelse_time}")

        # if len(body_list) > 0:
        #     body_list = self.body(body_list) # , cur_sample_size+len(body_list))
        #     res_list.extend(body_list)

        # if len(orelse_list) > 0:
        #     orelse_list = self.orelse(orelse_list) # , cur_sample_size+len(orelse_list))
        #     res_list.extend(orelse_list)

        # # print(f"length, res: {len(res_list)}")
        # # SAMPLING
        # res_symbol_table_list = adapt_sampling_distribution(res_list)
        # res_symbol_table_list = sample(res_symbol_table_list)
        # res_symbol_table_list, cur_sample_size = sample(res_symbol_table_list, cur_sample_size)

        # print(f"Result of IFelse: {[(res['x'].c, res['x'].delta) for res in res_list]}")
        # print(f"############end one ifelse ##################")
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
        # res_symbol_table_list = list()
        # counter = 0

        while(len(symbol_table_list) > 0):
            # counter += 1
            res_list = list()

            # before_calculate_branch = time.time()
            res_list = calculate_branch_list(self.target_idx, self.test, symbol_table_list)
            # print(f"calculate branch list time: {time.time() - before_calculate_branch}")
            # orelse_list= calculate_branch_list(self.target_idx, self.test, symbol_table_list, 'orelse')
            # print(f"length: {len(body_list)}, {len(orelse_list)}")

            # res_list.extend(body_list)
            # res_list.extend(orelse_list)

            # before_sample = time.time()
            res_list = adapt_sampling_distribution(res_list)
            res_list = sample(res_list)
            # print(f"sample in while: {time.time() - before_sample}")

            if res_list[0]['branch'] == 'body':
                # before_body = time.time()
                symbol_table_list = self.body(res_list)
                # print(f"run body: {time.time() - before_body}")
            else:
                # print(f"enter while {counter} times.")
                return res_list

            # SAMPLING the orelse partitions
        #     if len(orelse_list) > 0:
        #         tmp_res_symbol_table_list = adapt_sampling_distribution(orelse_list)
        #         tmp_res_symbol_table_list = sample(tmp_res_symbol_table_list)
        #         res_symbol_table_list.extend(tmp_res_symbol_table_list)
        #     # res_symbol_table_list.extend(orelse_list)
            
        #     if len(body_list) > 0:
        #         # print(f"test: {self.test}, target_idx: {self.target_idx}")
        #         symbol_table_list = self.body(body_list) # [body_list, cur_sample_size])
        #     else:
        #         symbol_table_list = list()

        # return res_symbol_table_list
        return res_list


def update_trajectory(symbol_table, target_idx):
    input = symbol_table['x'].select_from_index(0, target_idx)
    input_interval = input.getInterval()
    # print(f"input: {input.c, input.delta}")
    # print(f"input_interval: {input_interval.left.data.item(), input_interval.right.data.item()}")
    assert input_interval.left.data.item() <= input_interval.right.data.item()

    symbol_table['safe_range'].left = torch.min(symbol_table['safe_range'].left, input_interval.left)
    symbol_table['safe_range'].right = torch.max(symbol_table['safe_range'].right, input_interval.right)
    return symbol_table


class Trajectory(nn.Module):
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




            

            









