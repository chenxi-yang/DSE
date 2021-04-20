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


def show_tra_l(l):
    for abstract_state in l:
        print("in one abstract state")
        tra_len_l = list()
        for symbol_table in abstract_state:
            if len(symbol_table) > 1:
                tra_len_l.append(len(symbol_table['trajectory']))
                print(f"c: {symbol_table['x'].c},  delta: {symbol_table['x'].delta}")
            else:
                tra_len_l.append(0)
        # print(tra_len_l)

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

def calculate_abstract_state(target_idx, arg_idx, f, abstract_state):
    # assign_time = time.time()
    # if debug:
    #     r1 = torch.cuda.memory_reserved(0) 
    #     a1 = torch.cuda.memory_allocated(0)
    #     print("begin calculate_abstract_state")
    #     for i, symbol_table in enumerate(abstract_state):
    #         if len(symbol_table) > 0:
    #             print(i, symbol_table['x'].c)
    #             break
        
    for idx, symbol_table in enumerate(abstract_state):
        if len(symbol_table) == 0:
            abstract_state[idx] = symbol_table
            continue
        # print(symbol_table)
        x = symbol_table['x']
        input = x.select_from_index(0, arg_idx) # torch.index_select(x, 0, arg_idx)
        # print(f"f: {f}")
        # if debug:
        #     r3 = torch.cuda.memory_reserved(0) 
        #     a3 = torch.cuda.memory_allocated(0)
        #     print(f"memory before f: {a3}")
        #     print("before", f, input.c, input.delta)
        res  = f(input)
        # if debug:
        #     r4 = torch.cuda.memory_reserved(0) 
        #     a4 = torch.cuda.memory_allocated(0)
        #     print(f"memory after f: {a4}")
        #     print("after", f, res.c, res.delta)
        # print(f"calculate_x_list --  target_idx: {target_idx}, res: {res.c}, {res.delta}")
        x.set_from_index(target_idx, res) # x[target_idx[0]] = res
        # print("after set index", f, x.c, x.delta)
        symbol_table['x'] = x
        #! probability of each component does not change
        # symbol_table['probability'] = symbol_table['probability']
        abstract_state[idx] = symbol_table
    # print(f"-- assign -- calculate_x_list: {time.time() - assign_time}")
    # if debug:
    #     r2 = torch.cuda.memory_reserved(0) 
    #     a2 = torch.cuda.memory_allocated(0)
    #     if a2 > a1: 
    #         print(f"len of abstract_states: {len(abstract_state)}, close calculation, before, cuda memory reserved: {r1}, allocated: {a1}")
    #         print(f"func name: {f}")
    #         print(f"close calculation, after, cuda memory reserved: {r2}, allocated: {a2}")
    #     print("end calculate_abstract_state")
    #     for i, symbol_table in enumerate(abstract_state):
    #         if len(symbol_table) > 0:
    #             print(i, symbol_table['x'].c)
    #             break
    return abstract_state


def calculate_abstract_states_list(target_idx, arg_idx, f, abstract_state_list):
    # if debug:
    #     r = torch.cuda.memory_reserved(0) 
    #     a = torch.cuda.memory_allocated(0)
    #     print(f"calculate_abstract_states_list, before, cuda memory reserved: {r}, allocated: {a}")
    res_list = list()
    for abstract_state in abstract_state_list:
        res_abstract_state = calculate_abstract_state(target_idx, arg_idx, f, abstract_state)
        res_list.append(res_abstract_state)
    
    # if debug:
    #     r = torch.cuda.memory_reserved(0) 
    #     a = torch.cuda.memory_allocated(0)
    #     print(f"calculate_abstract_states_list, after, cuda memory reserved: {r}, allocated: {a}")
    return res_list


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
    # print(f"calculate branch -- target_idx: {target_idx}")
    # print(f"x: {x.c, x.delta}")

    x = symbol_table['x']
    target = x.select_from_index(0, target_idx)
    # res_symbol_table = pre_build_symbol_table(symbol_table)

    if target.getRight().data.item() <= test.data.item():
        res = x.clone()
        branch = 'body'
        body_symbol_table = pre_build_symbol_table(symbol_table)
        probability = pre_allocate(symbol_table) # the pobability represents the upper bound, so it does not change when splitting
        body_symbol_table = update_res_in_branch(body_symbol_table, res, probability, branch)
    elif target.getLeft().data.item() > test.data.item():
        res = x.clone()
        branch = 'orelse'
        orelse_symbol_table = pre_build_symbol_table(symbol_table)
        probability = pre_allocate(symbol_table)
        orelse_symbol_table = update_res_in_branch(orelse_symbol_table, res, probability, branch)
    else:
        # print(f"In split branch symbol table\n \
        #     x: {x.c, x.delta}\n  \
        #     target: {target.getLeft(), target.getRight()}\n \
        #     test: {test.data.item()}")
        # exit(0)
        res = x.clone()
        branch = 'body'
        c = (target.getLeft() + test) / 2.0
        delta = (test - target.getLeft()) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta)) # res[target_idx] = Box(c, delta)
        body_symbol_table = pre_build_symbol_table(symbol_table)
        # This is sound SE, so probability is the kept
        probability = pre_allocate(symbol_table)
        body_symbol_table = update_res_in_branch(body_symbol_table, res, probability, branch)

        res = x.clone()
        branch = 'orelse'
        c = (target.getRight() + test) / 2.0
        delta = (target.getRight() - test) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta))
        orelse_symbol_table = pre_build_symbol_table(symbol_table)

        probability = pre_allocate(symbol_table)
        orelse_symbol_table = update_res_in_branch(orelse_symbol_table, res, probability, branch)

    # print(f"branch time: {time.time() - branch_time}")
    return body_symbol_table, orelse_symbol_table


def abstract_state_empty(abstract_state):
    for symbol_table in abstract_state:
        if len(symbol_table) > 0:
            return False
    return True
            

def split_branch_abstract_state(target_idx, test, abstract_state):
    # if symbol_table is empty, keep it for sequence join
    body_abstract_state, orelse_abstract_state = list(), list()
    for symbol_table in abstract_state:
        # print(symbol_table)
        if len(symbol_table) == 0:
            body_symbol_table, orelse_symbol_table = dict(), dict()
        else:
            body_symbol_table, orelse_symbol_table = split_branch_symbol_table(target_idx, test, symbol_table)
        body_abstract_state.append(body_symbol_table)
        orelse_abstract_state.append(orelse_symbol_table)
    if abstract_state_empty(body_abstract_state):
        body_abstract_state = list()
    if abstract_state_empty(orelse_abstract_state):
        orelse_abstract_state = list()

    return body_abstract_state, orelse_abstract_state


'''
abstract_state:
list of symbol table with domain, probability
'''
def split_branch_list(target_idx, test, abstract_state_list):
    body_abstract_state_list, orelse_abstract_state_list = list(), list()
    for abstract_state in abstract_state_list:
        body_abstract_state, orelse_abstract_state = split_branch_abstract_state(target_idx, test, abstract_state)
        if len(body_abstract_state) > 0:
            body_abstract_state_list.append(body_abstract_state)
        if len(orelse_abstract_state) > 0:
            orelse_abstract_state_list.append(orelse_abstract_state)
    
    return body_abstract_state_list, orelse_abstract_state_list 


def sound_join_trajectory(trajectory_1, trajectory_2):
    l1, l2 = len(trajectory_1), len(trajectory_2)
    # if debug:
    #     print(f"Sound Join Trajectory: {l1}, {l2}")
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
    
    # if debug:
    #     print(f"Sound Join Trajectory End: {len(trajectory)}")
    assert(len(trajectory) >= max(l1, l2))

    return trajectory


def sound_join_symbol_table(symbol_table_1, symbol_table_2):
    symbol_table = {
        'x': symbol_table_1['x'].sound_join(symbol_table_2['x']),
        'probability': torch.max(symbol_table_1['probability'], symbol_table_2['probability']),
        'trajectory': sound_join_trajectory(symbol_table_1['trajectory'], symbol_table_2['trajectory']), 
        'branch': '',
    }
    return symbol_table


def sound_join_symbol_tables(symbol_tables):
    res_symbol_table = dict()
    for symbol_table in symbol_tables:
        if len(symbol_table) == 0: # if the symbol_table is empty, skip
            continue
        if len(res_symbol_table) == 0:
            res_symbol_table = symbol_table
        else:
            res_symbol_table = sound_join_symbol_table(res_symbol_table, symbol_table)
    return res_symbol_table


def sound_join_abstract_states(abstract_states):
    # list of abstract_states, join their symbol_tables sequentially
    # symbol_table might be empty, then just skip
    # symbol_table: {
    #   'x'
    #   'probability'
    #   'branch'
    #   'trajectory'
    # }
    new_abstract_state = list()
    number_symbol_table = len(abstract_states[0]) # all the number of symbol_tables should be the same
    for symbol_tables in list(zip(*abstract_states)):
        symbol_table = sound_join_symbol_tables(symbol_tables)
        new_abstract_state.append(symbol_table)
    return new_abstract_state


def sound_join_k(l1, l2, k):
    '''
    l1, l2: list of abstract states
    k: maximum allowed separate abstract_states
    '''
    # if debug:
    #     r = torch.cuda.memory_reserved(0) 
    #     a = torch.cuda.memory_allocated(0)
    #     print(f"sound_join_k, before, cuda memory reserved: {r}, allocated: {a}") 
        
    res_list = list()
    res_list.extend(l1)
    res_list.extend(l2)

    if len(res_list) <= k:
        return res_list

    shuffle(res_list)
    chunk_size = len(res_list) // k
    
    to_join_abstract_states_list = [res_list[i:i+chunk_size] for i in range(0, len(res_list), chunk_size)]
    res_list = list()
    for to_join_abstract_states in to_join_abstract_states_list:
        joined_abstract_state = sound_join_abstract_states(to_join_abstract_states)
        res_list.append(joined_abstract_state)
    
    # if debug:
    #     r = torch.cuda.memory_reserved(0) 
    #     a = torch.cuda.memory_allocated(0)
    #     print(f"sound_join_k, after, cuda memory reserved: {r}, allocated: {a}")
    
    return res_list


def batch_list(l1, l2):
    # batch the list together
    return l2
    

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
    
    def forward(self, abstract_state_list, cur_sample_size=0):
        # print(f"Assign Before: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        # print("In Assign")
        # for i, abstract_state in enumerate(abstract_state_list):
        #     for j, symbol_table in enumerate(abstract_state):
        #         if len(symbol_table) > 0:
        #             print(i, j, symbol_table['x'].c)
        #             break
        # if debug:
        #     print(f"before assign: {self.f}")
        #     show_tra_l(abstract_state_list)
        res_list = calculate_abstract_states_list(self.target_idx, self.arg_idx, self.f, abstract_state_list)
        # if debug:
        #     print(f"after assign: {self.f}")
        #     show_tra_l(res_list)
        # print(f"Assign After: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        # print("After Assign")
        # for i, abstract_state in enumerate(abstract_state_list):
        #     for j, symbol_table in enumerate(abstract_state):
        #         if len(symbol_table) > 0:
        #             print(i, j, symbol_table['x'].c)
        #             if symbol_table['x'].c.requires_grad:
        #                 exit(0)
        #             break
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
        test = self.f_test(self.test)
        res_list = list()
        # if debug:
        #     r = torch.cuda.memory_reserved(0) 
        #     a = torch.cuda.memory_allocated(0)
        #     print(f"split_branch_list, before, cuda memory reserved: {r}, allocated: {a}")
        # if debug:
        #     print(f"IfElse: ini")
        #     show_tra_l(abstract_state_list)
        body_list, else_list = split_branch_list(self.target_idx, self.test, abstract_state_list)
        # if debug:
        #     r = torch.cuda.memory_reserved(0) 
        #     a = torch.cuda.memory_allocated(0)
        #     print(f"split_branch_list, after, cuda memory reserved: {r}, allocated: {a}")
        # if debug:
            # print(f"IfElse: after split_branch_list, body_list:")
            # show_tra_l(body_list)
            # print(f"else_list:")
            # show_tra_l(else_list)

        if len(body_list) > 0:
            body_list = self.body(body_list)
            res_list = body_list
            # res_list.extend(body_list)
        if len(else_list) > 0:
            else_list = self.orelse(else_list)
            res_list = else_list
            # res_list.extend(else_list)
        # if debug:
        #     print(f"IfElse: before sound join k, body_list:")
        #     show_tra_l(body_list)
        #     print(f"else_list:")
        #     show_tra_l(else_list)
        # res_list = sound_join_k(body_list, else_list, k=constants.verification_num_abstract_states)
        # if debug:
        #     print(f"IfElse: after sound join k")
        #     show_tra_l(res_list)

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
    
    def forward(self, abstract_state_list):
        '''
        super set of E_{i-th step} and [\neg condition]
        '''
        # print(f"##############In while sound#########")
        i = 0
        res_list = list()
        while(len(abstract_state_list) > 0):
            # counter += 1
            # print("In  While", abstract_state_list[0][0]["x"].c)
            # if debug:
            #     print(f"in while: {i}")
            #     show_tra_l(abstract_state_list)

            body_list, else_list = split_branch_list(self.target_idx, self.test, abstract_state_list)
            # if debug:
            #     print(f"in while, body_list")
            #     show_tra_l(body_list)
            #     print(f"in while, else_list")
            #     show_tra_l(else_list)
            
            if len(else_list) > 0:
                # res_list.extend(else_list)
                # if debug:
                #     print(f"in while, before sound_join, res_list")
                #     show_tra_l(res_list)
                # res_list = sound_join_k(res_list, else_list, k=constants.verification_num_abstract_states)
                res_list = batch_list(res_list, else_list)
                # if debug:
                #     print(f"in while, after sound_join, res_list")
                #     show_tra_l(res_list)

            if len(body_list) > 0:
                abstract_state_list = self.body(body_list)
            else:
                return res_list
            i += 1
            # print(i, len(abstract_state_list),  len(res_list))
            # if debug:
            #     r = torch.cuda.memory_reserved(0) 
            #     a = torch.cuda.memory_allocated(0)
            #     print(f"while, cuda memory reserved: {r}, allocated: {a}")
            if i > 500:
                break
        res_list.extend(abstract_state_list)
        # if debug:
        #     print(f"end of while, break")
        #     show_tra_l(res_list)
        #     exit(0)

        return res_list


def update_trajectory(symbol_table, target_idx):
    # trajectory: list of states
    # states: list of intervals
    input_interval_list = list()
    for idx in target_idx:
        input = symbol_table['x'].select_from_index(0, idx)
        input_interval = input.getInterval()
        # print(f"input: {input.c, input.delta}")
        # print(f"input_interval: {input_interval.left.data.item(), input_interval.right.data.item()}")
        assert input_interval.left.data.item() <= input_interval.right.data.item()
        input_interval_list.append(input_interval)
    # print(f"In update trajectory")
    symbol_table['trajectory'].append(input_interval_list)
    # print(f"Finish update trajectory")

    return symbol_table


def update_abstract_state_trajectory(abstract_state, target_idx):
    for idx, symbol_table in enumerate(abstract_state):
        if len(symbol_table) == 0:
            pass
        else:
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




            

            









