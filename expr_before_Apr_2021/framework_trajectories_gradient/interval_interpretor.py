import torch
import random
from torch.autograd import Variable
import copy
import sys
import queue
import nlopt
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

import domain
from constants import *
from helper import *

def show(symbol_table):
    print('=======symbol table======')
    if symbol_table['probability'].data.item() <= 0.0:
        print('Not Exist')
        return 
    for symbol in symbol_table:
        if symbol == 'probability':
            print(symbol, symbol_table[symbol].data.item())
        else:
            print(symbol, symbol_table[symbol].left.data.item(), symbol_table[symbol].right.data.item())
    return 


def var(i, requires_grad=False):
    return Variable(torch.tensor(i, dtype=torch.float), requires_grad=requires_grad)


def run_next_stmt(next_stmt, symbol_table):
    if next_stmt is None:
        return symbol_table
    else:
        return next_stmt.execute(symbol_table)


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)

    return res_interval


def f_beta(beta):
    gamma = var(0.1)

    # return torch.min(var(0.5), beta)
    return beta


def pho(X, intersection):
    # partial_intersection_value = intersection.getVolumn().div(X.getVolumn())
    # if partial_intersection_value.data.item() < 0.5:
    #     partial_value = partial_intersection_value.div(f_beta(BETA))
    # else:
    #     partial_value = var(1.0).sub((var(1.0).sub(partial_intersection_value)).div(f_beta(BETA)))

    partial_value = intersection.getVolumn().div(X.getVolumn().mul(f_beta(INTERVAL_BETA)))
    # print('partial_value', partial_value)

    return torch.min(var(1.0), partial_value)


def update_symbol_table_with_constraint(target, test, symbol_table, direct):
    constraint_interval = domain.Interval(-10000.0, 10000.0)
    res_symbol_table = dict()
    # self-deepcopy symbol_table
    for symbol in symbol_table:
        # print(symbol)
        if symbol == 'probability':
            res_symbol_table[symbol] = var(symbol_table[symbol].data.item())
        else:
            # print(symbol)
            res_symbol_table[symbol] = domain.Interval(symbol_table[symbol].left.data.item(), symbol_table[symbol].right.data.item())
    
    target_value = res_symbol_table[target]
    # print('------in constraint')
    # show(symbol_table)
    # print('target-name', target)
    # print('test', test)
    # print('constraint', target_value.left, target_value.right, test)
    # print('direct', direct)
    # print('probability', symbol_table['probability'])

    if direct == '<':
        constraint_interval.right = test
    else:
        constraint_interval.left = test

    intersection_interval = get_intersection(target_value, constraint_interval)
    # print('intersection', intersection_interval.left, intersection_interval.right)

    if intersection_interval.isEmpty():
        intersection_interval = None
        probability = var(0.0)
    else:
        if target_value.isPoint():
            if direct == '<' and target_value.right.data.item() <= constraint_interval.right.data.item():
                probability = symbol_table['probability']
            elif direct == '>' and target_value.left.data.item() > constraint_interval.left.data.item():
                probability = symbol_table['probability']
            else:
                intersection_interval = None
                probability = var(0.0)
        elif intersection_interval.left.data.item() == intersection_interval.right.data.item() and (intersection_interval.left.data.item() == constraint_interval.left.data.item() or intersection_interval.right.data.item() == constraint_interval.right.data.item()):
            intersection_interval = None
            probability = var(0.0)
        else:
            # print('beta', f_beta(BETA))
            # print('pho', pho(target_value, intersection_interval))
            probability = symbol_table['probability'].mul(pho(target_value, intersection_interval))

    res_symbol_table[target] = intersection_interval
    res_symbol_table['probability'] = probability
    # print('probability', res_symbol_table['probability'])
    # # print('final intersection', intersection_interval)
    # show(res_symbol_table)
    # print('------out constraint')

    return res_symbol_table


def update_symbol_table(target, func, symbol_table):
    # print('----before assign')
    # show(symbol_table)

    #! assume only monotone function
    # target_value = symbol_table[target[0]]
    res_target = target[0]
    value_length = len(target)
    left_value = P_INFINITY
    right_value = N_INFINITY

    tmp_value_list = list()
    tmp_idx_list = list()
    def generate_idx(value_length, str=''):
        if len(str) == value_length:
            tmp_idx_list.append(str)
            return 
        for digit in '012':
            generate_idx(value_length, str+digit)
    
    generate_idx(value_length)
    
    for idx_guide in tmp_idx_list:
        value_list = list()
        for idx, i in enumerate(idx_guide):
            if i == '0':
                value_list.append(symbol_table[target[idx]].left)
            elif i == '1':
                value_list.append(symbol_table[target[idx]].right)
            else:
                if symbol_table[target[idx]].left.data.item() < 0.0 and symbol_table[target[idx]].left.data.item() > 0.0:
                    value_list.append(var(0.0))
                else:
                    value_list.append(symbol_table[target[idx]].right)
        tmp_value_list.append(value_list)
    
    tmp_res_value_list = [func(value_list) for value_list in tmp_value_list]

    for tmp_res in tmp_res_value_list:
        left_value = torch.min(left_value, tmp_res)
        right_value = torch.max(right_value, tmp_res)

    symbol_table[res_target].left = left_value
    symbol_table[res_target].right = right_value

    # value_left_list = [symbol_table[symbol].left for symbol in target]
    # value_right_list = [symbol_table[symbol].right for symbol in target]

    # symbol_table[res_target].left = func(value_left_list)
    # symbol_table[res_target].right = func(value_right_list)

    # symbol_table[res_target].left = func(target_value.left)
    # symbol_table[res_target].right = func(target_value.right)

    # print('in update symbol table --- assign')
    # show(symbol_table)


    return symbol_table


def get_overapproximation(interval1, interval2):
    res_interval = domain.Interval()
    res_interval.left = torch.min(interval1.left, interval2.left)
    res_interval.right = torch.max(interval1.right, interval2.right)

    return res_interval 


def join(symbol_table_1, symbol_table_2):
    #! assume only one symbol change each time

    target_symbol_list = list()
    symbol_table = dict()
    for symbol in symbol_table_1:
        if symbol == 'probability':
            continue

        if symbol_table_1[symbol] is None or (not symbol_table_1[symbol].equal(symbol_table_2[symbol])):
                target_symbol_list.append(symbol)
                continue
        symbol_table[symbol] = symbol_table_1[symbol]
    
    # print('isOn')
    # print(symbol_table_1['isOn'].left, symbol_table_1['isOn'].right)
    # print(symbol_table_2['isOn'].left, symbol_table_2['isOn'].right)
    
    p1 = symbol_table_1['probability']
    p2 = symbol_table_2['probability']

    if p1.data.item() <= 0.0:
        return symbol_table_2
    if p2.data.item() <= 0.0:
        return symbol_table_1
    # print('p1, p2', p1, p2)

    p_out = torch.min(p1.add(p2), var(1.0))
    
    for target_symbol in target_symbol_list:
        
        target1 = symbol_table_1[target_symbol]
        target2 = symbol_table_2[target_symbol]

        # print('target', target_symbol)
        # print(target1.left, target1.right)
        # print(target2.left, target2.right)

        if target1 is None:
            symbol_table[target_symbol] = target2
            continue
        elif target2 is None:
            symbol_table[target_symbol] = target1
            continue

        c1 = target1.getCenter()
        c2 = target2.getCenter()

        c_out = (p1.mul(c1)).add(p2.mul(c2)).div(p1.add(p2))

        p1_prime = p1.div(torch.max(p1, p2))
        p2_prime = p2.div(torch.max(p1, p2))

        c1_prime = p1_prime.mul(c1).add((var(1.0).sub(p1_prime)).mul(c_out))
        c2_prime = p2_prime.mul(c2).add((var(1.0).sub(p2_prime)).mul(c_out))

        l1_prime = p1_prime.mul(target1.getLength())
        l2_prime = p2_prime.mul(target2.getLength())

        if target1.left.data.item() == target1.right.data.item() and target2.left.data.item() == target2.right.data.item():
            target1.left = c1_prime
            target1.right = c1_prime
            target2.left = c2_prime
            target2.right = c2_prime
        else:
            target1.left = c1_prime.sub(l1_prime.div(var(2.0)))
            target1.right = c1_prime.add(l1_prime.div(var(2.0)))
            target2.left = c2_prime.sub(l2_prime.div(var(2.0)))
            target2.right = c2_prime.add(l2_prime.div(var(2.0)))
        # print('Join')
        # print('target1', target1.left, target1.right)
        # print('target2', target2.left, target2.right)

        symbol_table[target_symbol] = get_overapproximation(target1, target2)
    
    symbol_table['probability'] = p_out

    # print('=================end join================')

    return symbol_table


class Ifelse:
    def __init__(self, target, test, body, orelse, next_stmt):
        self.target = target
        self.test = test
        self.body = body
        self.orelse = orelse
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):
        # print('--- in ifelse target, test', self.target, self.test)
        # show(symbol_table)

        this_block_p = symbol_table['probability'].data.item()
        body_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '<')
        orelse_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '>')
        # print('after if else split', self.target)
        # print('probability', body_symbol_table['probability'], orelse_symbol_table['probability'])
        # print('--body')
        # show(body_symbol_table)
        # print('--orelse')
        # show(orelse_symbol_table)


        if body_symbol_table['probability'].data.item() > 0.0:
            body_symbol_table = self.body.execute(body_symbol_table)
        # else:
        #     return run_next_stmt(self.next_stmt, orelse_symbol_table)
        
        # print('a show1')
        # print('--1--body')
        # show(body_symbol_table)
        # print('--1--orelse')
        # show(orelse_symbol_table)
        # print('b show1')
        
        if orelse_symbol_table['probability'].data.item() > 0.0:
            orelse_symbol_table = self.orelse.execute(orelse_symbol_table)
        # else:
        #     return run_next_stmt(self.next_stmt, body_symbol_table)

        # print('a show2')
        # print('--2--body')
        # show(body_symbol_table)
        # print('--2--orelse')
        # show(orelse_symbol_table)
        # print('b show2')

        symbol_table = join(body_symbol_table, orelse_symbol_table)
        symbol_table['probability'] = var(this_block_p)

        # print('--after join')
        # show(symbol_table)

        # print('---end if')

        return run_next_stmt(self.next_stmt, symbol_table)


class WhileSimple:
    #! not a real loop, just in the form of loop to operate several if-else stmt
    def __init__(self, target, test, body, next_stmt):
        # TODO: implement while & test
        self.target = target
        self.test = test
        self.body = body
        self.next_stmt = next_stmt
     
    def execute(self, symbol_table):

        this_block_p = symbol_table['probability'].data.item()
        
        body_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '<')
        # orelse_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '>')

        i_list = list()
        l_list = list()
        r_list = list()

        i_list.append(symbol_table['i'].left.data.item())
        l_list.append(symbol_table['x'].left.data.item())
        r_list.append(symbol_table['x'].right.data.item())
        # print('11111, pro', body_symbol_table['probability'].data.item(), (var(1.0).div(f_beta(BETA))).data.item())

        #! if body_symbol_table['probability'].data.item() == (var(1.0).div(f_beta(BETA))).data.item():
        #     print('111-----------------------------this block p', this_block_p)
        body_symbol_table['probability'] = var(1.0)

        while(body_symbol_table['probability'].data.item() > 0):
            # print('probability', body_symbol_table['probability'].data.item())
            # print('---in beginning while block , target', self.target)
            # show(body_symbol_table)
            #! do not propogate probability too much -- f(beta)
            # if body_symbol_table['probability'].data.item() == (var(1.0).div(f_beta(BETA))).data.item():
            #     print('-----------------------------this block p', this_block_p)
            body_symbol_table['probability'] = var(1.0)

            body_symbol_table = self.body.execute(body_symbol_table)
            symbol_table = body_symbol_table

            i_list.append(symbol_table['i'].left.data.item())
            l_list.append(symbol_table['x'].left.data.item())
            r_list.append(symbol_table['x'].right.data.item())

            # print('interval', symbol_table['x'].left.data.item(), symbol_table['x'].right.data.item())
            # print('after execute')
            # show(body_symbol_table)

            # print('before check i value')
            body_symbol_table = update_symbol_table_with_constraint(self.target, self.test, body_symbol_table, '<')
            # print(symbol_table[self.target].left, symbol_table[self.target].right)
            # print('after check i value')
            # print('--in while')
            # show(symbol_table)
            # print('---in end while block')
            # orelse_symbol_table = update_symbol_table_with_constraint(self.target, self.test, copy.deepcopy(tmp_symbol_table), '>')

        # for i, value in enumerate(l_list):
        #     print(l_list[i], r_list[i])
        # plt.plot(i_list, l_list, label="left", marker='x')
        # plt.plot(i_list, r_list, label="right", marker ='x')
        # plt.xlabel('loop')
        # plt.ylabel('l-r')
        # plt.legend()
        # plt.show()

        return run_next_stmt(self.next_stmt, symbol_table)


class Assign:
    def __init__(self, target, value, next_stmt):
        self.target = target
        self.value = value
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):
        symbol_table = update_symbol_table(self.target, self.value, symbol_table)

        return run_next_stmt(self.next_stmt, symbol_table)