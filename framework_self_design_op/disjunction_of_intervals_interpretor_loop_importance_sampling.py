import torch
import random
from random import shuffle
from torch.autograd import Variable
import copy
import sys
import queue
import nlopt
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from timeit import default_timer as timer

import domain
from constants import *
from helper import *

target = domain.Interval(safe_l, safe_r)

def run_next_stmt(next_stmt, symbol_table, cur_sample_size):
    if next_stmt is None:
        return symbol_table
    else:
        return next_stmt.execute(symbol_table, cur_sample_size)

def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)

    return res_interval


def distance_interval(X, target):
    # if X.right.data.item() > 15.0:
    #     print('in distance interval')
    #     print(X.right.data.item(), target.right.data.item())
    # if X.right.data.item() > 11.0:
    #     print('X: ', X.left.data.item(), X.right.data.item())
    #     print('target: ', target.left.data.item(), target.right.data.item())
    intersection_interval = get_intersection(X, target)
    if intersection_interval.isEmpty():
        res = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
    elif intersection_interval.getLength().data.item() == X.getLength().data.item():
        if target.left.data.item() == N_INFINITY:
            res = var(0.0).sub(target.right.sub(X.right))
        elif target.right.data.item() == P_INFINITY:
            res = var(0.0).sub(X.left.sub(target.left))
        else:
            res = var(0.0).sub(target.getLength().sub(X.getLength()))
    else:
        res = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
    # if X.right.data.item() > 11.0:
    #     print('result: ', res.data.item())
    return res


def f_beta(beta):
    gamma = var(0.1)
    # return torch.min(var(0.5), beta)
    return beta


def pho(X, intersection):
    partial_intersection_value = intersection.getVolumn().div(X.getVolumn())

    return torch.min(var(1.0), partial_intersection_value)


# TODO: adapt to different domains
def update_symbol_table_with_constraint(target, test, symbol_table, direct):
    constraint_interval = domain.Interval(-10000.0, 10000.0)
    res_symbol_table = dict()
    # self-deepcopy symbol_table
    for symbol in symbol_table:
        if 'probability' in symbol:
            res_symbol_table[symbol] = var(symbol_table[symbol].data.item())
        elif 'list' in symbol:
            res_symbol_table[symbol] = list()
            for i in symbol_table[symbol]:
                res_symbol_table[symbol].append(i)
        else:
            res_symbol_table[symbol] = domain.Interval(symbol_table[symbol].left.data.item(), symbol_table[symbol].right.data.item())
    
    target_value = res_symbol_table[target]

    if direct == '<':
        constraint_interval.right = test
    else:
        constraint_interval.left = test

    intersection_interval = get_intersection(target_value, constraint_interval)
    # print('intersection', intersection_interval.left, intersection_interval.right)

    if intersection_interval.isEmpty():
        intersection_interval = None
        probability = var(0.0)
        # return None
    else:
        if target_value.isPoint():
            if direct == '<' and target_value.right.data.item() <= constraint_interval.right.data.item():
                probability = symbol_table['probability']
            elif direct == '>' and target_value.left.data.item() > constraint_interval.left.data.item():
                probability = symbol_table['probability']
            else:
                intersection_interval = None
                probability = var(0.0)
                # return None
        elif intersection_interval.left.data.item() == intersection_interval.right.data.item() and (intersection_interval.left.data.item() == constraint_interval.left.data.item() or intersection_interval.right.data.item() == constraint_interval.right.data.item()):
            intersection_interval = None
            probability = var(0.0)
            # return None
        else:
            probability = symbol_table['probability'].mul(pho(target_value, intersection_interval))

    res_symbol_table[target] = intersection_interval
    res_symbol_table['probability'] = probability
    res_symbol_table['explore_probability'] = res_symbol_table['probability']

    return res_symbol_table


def update_symbol_table_list_with_constraint(target, test, symbol_table_list, direct):
    res_symbol_table_list = list()
    for symbol_table in symbol_table_list:
        # print('XXXXXXXXXXXX before update symbol table with constraint')
        res_symbol_table = update_symbol_table_with_constraint(target, test, symbol_table, direct)
        # print('XXXXXXXXXXXX after update symbol table with constraint')
        if res_symbol_table is None:
            continue
        res_symbol_table_list.append(res_symbol_table)
    # print('update constraint')
    # show_symbol_tabel_list(res_symbol_table_list)
    return res_symbol_table_list


def update_symbol_table_before(target, func, symbol_table):
    # print('----before assign', target)
    # print(symbol_table)

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
                # print(idx)
                value_list.append(symbol_table[target[idx]].left)
            elif i == '1':
                value_list.append(symbol_table[target[idx]].right)
            else:
                if symbol_table[target[idx]].left.data.item() < 0.0 and symbol_table[target[idx]].right.data.item() > 0.0:
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

    return symbol_table


def check_eql_var(x, y):
    if torch.abs(x.sub(y)).data.item() < EPSILON.data.item():
        return True
    else:
        return False

# TODO: should adapt to other domains
def update_symbol_table(target, func, symbol_table):
    res_target = target[0]
    
    instance_list = [symbol_table[symbol] for symbol in target]
    symbol_table[res_target] = func(instance_list)

    return symbol_table


def show(symbol_table):
    # print('=======symbol table======')
    if symbol_table['probability'].data.item() <= 0.0:
        print('Not Exist')
        return 
    for symbol in symbol_table:
        if symbol == 'probability':
            print(symbol, symbol_table[symbol].data.item())
        else:
            print(symbol, symbol_table[symbol].left.data.item(), symbol_table[symbol].right.data.item())
    return 


def show_symbol_tabel_list(symbol_table_list):
    # TODO: should adapt to other domains
    print('symbol table list:', len(symbol_table_list))
    # l_min = 100000
    # r_max = -10000
    for symbol_table in symbol_table_list:
        if symbol_table['stage'] is None: 
            continue
        p = symbol_table['probability'].data.item()
        stage = symbol_table['stage'].left.data.item()
        l = symbol_table['x_min'].left.data.item()
        r = symbol_table['x_max'].right.data.item()
        # i = symbol_table['i'].right.data.item()
        isOn_l = symbol_table['u'].left.data.item()
        isOn_r = symbol_table['u'].right.data.item()

        print('stage: ', str(stage))
        print('probability: ' + str(p) + ', interval: ' + str(l) + ',' + str(r)  + ', x1: ' + str(isOn_l) + ',' + str(isOn_r))


def divide_list(l, k):
    avg = len(l) / float(k)
    res = list()
    last = 0.0

    while last < len(l):
        res.append(l[int(last):int(last + avg)])
        last += avg
    
    return res


def get_overapproximation_list(interval_list):
    # TODO: should adapt to other domains
    res_interval = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())

    for interval in interval_list:
        res_interval.left = torch.min(res_interval.left, interval.left)
        res_interval.right = torch.max(res_interval.right, interval.right)
    
    return res_interval


def join_list(symbol_table_list):
    # TODO: should adapt to other domains
    # print('=========== join list ===========')
    # show_symbol_tabel_list(symbol_table_list)
    target_symbol_list = list()
    res_symbol_table = dict()

    first_symbol_table = symbol_table_list[0]
    for symbol in first_symbol_table:
        if 'probability' in symbol:
            continue
        if all([first_symbol_table[symbol].equal(symbol_table[symbol]) for symbol_table in symbol_table_list]):
            res_symbol_table[symbol] = first_symbol_table[symbol]
        else:
            target_symbol_list.append(symbol)
    
    # update p, p_out
    p_list = [symbol_table['probability'] for symbol_table in symbol_table_list]
    p_out = var(0.0)
    p_max = N_INFINITY
    for p in p_list:
        p_out =  p_out.add(p)
        p_max = torch.max(p_max, p)
    p_out = torch.min(p_out, var(1.0))
    p_prime_list = [p.div(p_max) for p in p_list]

    for symbol in target_symbol_list:
        target_list = [symbol_table[symbol] for symbol_table in symbol_table_list]

        c_list = [target.getCenter() for target in target_list]
        c_out = var(0.0)
        for idx, c in enumerate(c_list):
            p = p_list[idx]
            c_out = c_out.add(p.mul(c))
        c_out = c_out.div(p_out)
        
        c_prime_list = list()
        l_prime_list = list()
        for idx, target in enumerate(target_list):
            c = c_list[idx]
            p_prime = p_prime_list[idx]

            c_prime_list.append(p_prime.mul(c).add((var(1.0).sub(p_prime)).mul(c_out)))
            l_prime_list.append(p_prime.mul(target.getLength()))
        
        if all([target.isPoint() for target in target_list]):
            for idx, target in enumerate(target_list):
                target.left = c_prime_list[idx]
                target.right = c_prime_list[idx]
                target_list[idx] = target
        else:
            for idx, target in enumerate(target_list):
                if target.isPoint():
                    target.left = c_prime_list[idx]
                    target.right = c_prime_list[idx]
                else:
                    target.left = c_prime_list[idx].sub(l_prime_list[idx].div(var(2.0)))
                    target.right = c_prime_list[idx].add(l_prime_list[idx].div(var(2.0)))
                target_list[idx] = target
        
        res_symbol_table[symbol] = get_overapproximation_list(target_list)
    res_symbol_table['probability'] = p_out

    # define the explore probability
    res_symbol_table['explore_probability'] = res_symbol_table['probability']
    # show_symbol_tabel_list([res_symbol_table])
    # print('===========end join list===========')
    
    return res_symbol_table


def check_sampling(symbol_table):
    sample_flag = False

    probability = symbol_table['explore_probability']
    sample_flag = random.random() < probability.data.item()
    # print('check sampling:', probability, sample_flag)

    return sample_flag


def get_score_gradient(x_list, x, target):
    # TODO: adpat to different domains
    # print('score gradient', x.right.data.item(), x_list[0].right.data.item())
    # if x.right.data.item() > 15.0:
    #     print(x.right, target.right)
    score = var(0.0)
    x_score = distance_interval(x, target)
    # print('before loop')
    for x_interval in x_list:
        score = score.add(x_score.sub(distance_interval(x_interval, target)))
    score.div(var(len(x_list)))

    return score


def adapt_sampling_distribution(res_symbol_table_list):
    # print('before', res_symbol_table_list[0]['explore_probability'])
    # print('max', max_explore_probability)
    # print('after', res_symbol_table_list[0]['explore_probability'])
    
    if SAMPLE_METHOD == 3:
        length = len(res_symbol_table_list)
        tmp_explore_probability = res_symbol_table_list[0]['explore_probability']
        
        for idx, res_symbol_table in enumerate(res_symbol_table_list):
            re_idx = (idx - c) % length
            if re_idx == 0:
                res_symbol_table_list[idx]['explore_probability'] = tmp_explore_probability
            else:
                res_symbol_table_list[idx]['explore_probability'] = res_symbol_table_list[re_idx]['explore_probability']
    
    if SAMPLE_METHOD == 4:
        # print('enter adapt_samping_distribution, length', len(res_symbol_table_list))
        length = len(res_symbol_table_list)
        score_list = list()
        for res_symbol_table in res_symbol_table_list:
            x = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
            # TODO: should adapt to different domains
            x.left = torch.min(res_symbol_table['x_min'].left, res_symbol_table['x_max'].left)
            x.right = torch.max(res_symbol_table['x_min'].right, res_symbol_table['x_max'].right)
            score_list.append(get_score_gradient(res_symbol_table['x_memo_list'], x, target))
        # print('score list', score_list)
        score_idx_list = [x for x, y in sorted(enumerate(score_list), key = lambda x:x[1].data.item(), reverse=True)]
        probability_idx_list = [x for x, y in sorted(enumerate(res_symbol_table_list), key = lambda x:x[1]['explore_probability'].data.item(), reverse=True)]
        for idx, score_idx in enumerate(score_idx_list):
            # print(score_list[score_idx], res_symbol_table_list[probability_idx_list[idx]]['explore_probability'])
            res_symbol_table_list[score_idx]['explore_probability'] = res_symbol_table_list[probability_idx_list[idx]]['explore_probability']
        
    return res_symbol_table_list
        

def sample(symbol_table_list, cur_sample_size):
    start_t = timer()
    shuffle(symbol_table_list)

    res_symbol_table_list = list()
    to_get_sample_size = min(len(symbol_table_list), SAMPLE_SIZE - cur_sample_size)
    symbol_table_idx = 0

    max_explore_probability = N_INFINITY
    for symbol_table in symbol_table_list:
        max_explore_probability = torch.max(symbol_table['explore_probability'], max_explore_probability)
    
    for idx, symbol_table in enumerate(symbol_table_list):
        symbol_table_list[idx]['explore_probability'] = symbol_table_list[idx]['explore_probability'].div(max_explore_probability)

    while to_get_sample_size > 0:
        symbol_table_idx = symbol_table_idx%len(symbol_table_list)
        symbol_table = symbol_table_list[symbol_table_idx]
        if check_sampling(symbol_table):
            res_symbol_table_list.append(symbol_table)
            cur_sample_size += 1
            to_get_sample_size -= 1
            del symbol_table_list[symbol_table_idx]
        else:
            symbol_table_idx += 1
    
    for idx, symbol_table in enumerate(res_symbol_table_list):
        res_symbol_table_list[idx]['explore_probability'] = res_symbol_table_list[idx]['explore_probability'].mul(max_explore_probability)

    # print("---Sampling Length---" + str(len(res_symbol_table_list)))
    # print("--- %s seconds ---" % (timer() - start_t))

    return res_symbol_table_list, cur_sample_size
    

class Ifelse:
    def __init__(self, target, test, f, body, orelse, next_stmt):
        self.target = target
        self.test = test
        self.body = body
        self.orelse = orelse
        self.next_stmt = next_stmt
        self.f = f
    
    def execute(self, symbol_table_list, cur_sample_size=0):
        # print('In Ifelse, target, test: ', self.target, self.f(self.test).data.item())
        # show_symbol_tabel_list(symbol_table_list)
        num_disjunction = len(symbol_table_list)
        if cur_sample_size > 0:
            cur_sample_size -= num_disjunction

        res_symbol_table_list = list()
        body_symbol_table_list = update_symbol_table_list_with_constraint(self.target, self.f(self.test), symbol_table_list, '<')
        orelse_symbol_table_list = update_symbol_table_list_with_constraint(self.target, self.f(self.test), symbol_table_list, '>')

        tmp_body_symbol_table_list = list()
        for body_symbol_table in body_symbol_table_list:
            if body_symbol_table['probability'].data.item() > 0.0:
                tmp_body_symbol_table_list.append(body_symbol_table)
        
        # print('body, ', len(tmp_body_symbol_table_list))
            
        if len(tmp_body_symbol_table_list) > 0:
            body_symbol_table_list = self.body.execute(tmp_body_symbol_table_list, cur_sample_size+len(tmp_body_symbol_table_list))
            res_symbol_table_list.extend(body_symbol_table_list)
        
        tmp_orelse_symbol_table_list = list()
        for orelse_symbol_table in orelse_symbol_table_list:
            if orelse_symbol_table['probability'].data.item() > 0.0:
                tmp_orelse_symbol_table_list.append(orelse_symbol_table)
        # print('orelse, ', len(tmp_orelse_symbol_table_list))

        if len(tmp_orelse_symbol_table_list) > 0:
            orelse_symbol_table_list = self.orelse.execute(tmp_orelse_symbol_table_list, cur_sample_size+len(tmp_orelse_symbol_table_list))
            res_symbol_table_list.extend(orelse_symbol_table_list)
        
        # print('before sampling, cur sample size', cur_sample_size)
        # show_symbol_tabel_list(res_symbol_table_list)
        res_symbol_table_list = adapt_sampling_distribution(res_symbol_table_list)
        # print('middle sample, length', len(res_symbol_table_list), 'current sample size: ', cur_sample_size)
        res_symbol_table_list, cur_sample_size = sample(res_symbol_table_list, cur_sample_size)
        
        # print('after sampling, cur sample size', cur_sample_size)
        # show_symbol_tabel_list(res_symbol_table_list)
        # print('after sample, length', len(res_symbol_table_list), 'current sample size: ', cur_sample_size)

        if len(res_symbol_table_list) > K_DISJUNCTS:
            # print('in join, table length', len(res_symbol_table_list))
            symbol_table_list_to_join_list = divide_list(res_symbol_table_list, K_DISJUNCTS)
            res_symbol_table_list = list()
            for symbol_table_list_to_join in symbol_table_list_to_join_list:
                symbol_table_list_after_join = join_list(symbol_table_list_to_join)
                res_symbol_table_list.append(symbol_table_list_after_join)

        return run_next_stmt(self.next_stmt, res_symbol_table_list, cur_sample_size)


class WhileSimple:
    #! not a real loop, just in the form of loop to operate several if-else stmt
    def __init__(self, target, test, body, next_stmt):
        # TODO: implement while & test
        self.target = target
        self.test = test
        self.body = body
        self.next_stmt = next_stmt
     
    def execute(self, symbol_table_list, cur_sample_size=0):
        if cur_sample_size > 0:
            cur_sample_size -= len(symbol_table_list)

        this_block_p = symbol_table_list[0]['probability'].data.item()
        
        body_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table_list[0], '<')
        # orelse_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '>')

        body_symbol_table[0]['probability'] = var(1.0)

        symbol_table_queue = queue.Queue()
        symbol_table_queue.put(body_symbol_table[0])

        res_symbol_table_list = list()

        while(not symbol_table_queue.empty()):
            body_symbol_table = symbol_table_queue.get()

            if body_symbol_table['probability'].data.item() > 0:
            
                body_symbol_table_list = self.body.execute([body_symbol_table])

                # print('------------in while')
                # show_symbol_tabel_list(body_symbol_table_list)
                for tmp_body_symbol_table in body_symbol_table_list:
                    idx = int(tmp_body_symbol_table['i'].left.data.item())
                    l_value = tmp_body_symbol_table['x'].left.data.item()
                    r_value = tmp_body_symbol_table['x'].right.data.item()

                    # l_list[idx] = min(l_list[idx], l_value)
                    # r_list[idx] = max(r_list[idx], r_value)

                flag = 0.0

                for tmp_body_symbol_table in body_symbol_table_list:
                    new_tmp_body_symbol_table_list = update_symbol_table_with_constraint(self.target, self.test, tmp_body_symbol_table, '<')
                    if new_tmp_body_symbol_table_list[0]['probability'].data.item() <= 0.0:
                        flag = 1.0
                        break
                    symbol_table_queue.put(new_tmp_body_symbol_table_list[0])
                    # print('i', tmp_body_symbol_table['i'].left)
                
                if flag == 1.0:
                    for tmp_body_symbol_table in body_symbol_table_list:
                        res_symbol_table_list.append(tmp_body_symbol_table)
            
        return run_next_stmt(self.next_stmt, res_symbol_table_list, cur_sample_size)


class Assign:
    def __init__(self, target, value, next_stmt):
        self.target = target
        self.value = value
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table_list, cur_sample_size=0):
        if cur_sample_size > 0:
            cur_sample_size -= len(symbol_table_list)
        # print('assign', symbol_table_list)
        for idx, symbol_table in enumerate(symbol_table_list):
            if self.target == 'x':
                if len(symbol_table_list[idx]['x_memo_list']) >= k:
                    #TODO: adapt to all domains
                    x = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
                    x.left = torch.min(res_symbol_table['x_min'].left, res_symbol_table['x_max'].left)
                    x.right = torch.max(res_symbol_table['x_min'].right, res_symbol_table['x_max'].right)
                    symbol_table_list[idx]['x_memo_list'].attend(x)
                    del symbol_table_list[idx]['x_memo_list'][0]

            symbol_table_list[idx] = update_symbol_table(self.target, self.value, symbol_table)

        return run_next_stmt(self.next_stmt, symbol_table_list, cur_sample_size + len(symbol_table_list))


class WhileSample:
    def __init__(self, target, test, body, next_stmt):
        self.target = target
        self.test = test
        self.body = body
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table_list, cur_sample_size=0):
        num_disjunction = len(symbol_table_list)
        if cur_sample_size > 0:
            cur_sample_size -= len(symbol_table_list)
        res_symbol_table_list = list()
        # print('In WhileSample')

        # Protection Mechanism
        count = 0
        end_flag = 0

        while len(symbol_table_list) > 0:
            # show_symbol_tabel_list(res_symbol_table_list)
            # print('target', self.target)
            # # print('disjunction K: ', len(symbol_table_list))
            # show_symbol_tabel_list(symbol_table_list)
            path_probability_list = [symbol_table['probability'] for symbol_table in symbol_table_list]

            body_symbol_table_list = update_symbol_table_list_with_constraint(self.target, self.test, symbol_table_list, '<') # '<='
            # print('body symbol table')
            # show_symbol_tabel_list(body_symbol_table_list)
            orelse_symbol_table_list = update_symbol_table_list_with_constraint(self.target, self.test, symbol_table_list, '>')
            # print('body symbol table')
            # show_symbol_tabel_list(body_symbol_table_list)
            # print('orelse symbol table')
            # show_symbol_tabel_list(orelse_symbol_table_list)

            count_body = 0
            for body_symbol_table in body_symbol_table_list:
                if body_symbol_table['probability'].data.item() > 0.0:
                    count_body += 1
            count_orelse = 0
            for orelse_symbol_table in orelse_symbol_table_list:
                if orelse_symbol_table['probability'].data.item() > 0.0:
                    count_orelse += 1
            # print('Number back to loop', count_body)
            # print('Number out of loop', count_orelse)

            del_idx = 0
            tmp_res_symbol_table_list = list()

            for idx, orelse_symbol_table in enumerate(orelse_symbol_table_list):
                # print('len res', len(res_symbol_table_list))
                # print(len(res_symbol_table_list) * 1.0 / num_disjunction)
                if orelse_symbol_table['probability'].data.item() > 0.0:
                    if (body_symbol_table_list[idx - del_idx]['probability'].data.item() <= 0.0): #  or (len(res_symbol_table_list) * 1.0 / num_disjunction >= SAMPLE_SIZE):
                        # print(len(res_symbol_table_list) * 1.0 / num_disjunction)
                        # print('in del', body_symbol_table_list[idx - del_idx]['probability'].data.item(), len(res_symbol_table_list))
                        # orelse_symbol_table['probability'] = path_probability_list[idx]
                        # print(len(body_symbol_table_list))
                        # print(idx, del_idx)
                        del body_symbol_table_list[idx - del_idx]
                        del_idx += 1
                    # Sampling Condition
                    tmp_res_symbol_table_list.append(orelse_symbol_table)

                    # # if (len(res_symbol_table_list) * 1.0 / num_disjunction < SAMPLE_SIZE) or check_sampling(orelse_symbol_table):
                    # if cur_sample_size < SAMPLE_SIZE or check_sampling(orelse_symbol_table):
                    #     res_symbol_table_list.append(orelse_symbol_table)
                    #     cur_sample_size += 1
                    # # if check_sampling(orelse_symbol_table):
                    # # if len(res_symbol_table_list) * 1.0 / num_disjunction >= SAMPLE_SIZE:
                    # #     end_flag = 1
                    # #     break
            
            if len(tmp_res_symbol_table_list) > 0:
                tmp_res_symbol_table_list= adapt_sampling_distribution(tmp_res_symbol_table_list)
                tmp_res_symbol_table_list, cur_sample_size = sample(tmp_res_symbol_table_list, cur_sample_size)
                res_symbol_table_list.extend(tmp_res_symbol_table_list)
            
            # show res symbol table list
            # show_symbol_tabel_list(res_symbol_table_list)
            
            # smooth join 
            if len(res_symbol_table_list) > K_DISJUNCTS:
                symbol_table_list_to_join_list = divide_list(res_symbol_table_list, K_DISJUNCTS)
                res_symbol_table_list = list()
                for symbol_table_list_to_join in symbol_table_list_to_join_list:
                    symbol_table_list_after_join = join_list(symbol_table_list_to_join)
                    res_symbol_table_list.append(symbol_table_list_after_join)

            # if end_flag == 1:
            #     break
            
            if len(body_symbol_table_list) > 0:
                # print('body len', len(body_symbol_table_list))
                # body_symbol_table_list = adapt_sampling_distribution(body_symbol_table_list)
                # body_symbol_table_list, cur_sample_size = sample(body_symbol_table_list, cur_sample_size)
                symbol_table_list = self.body.execute(body_symbol_table_list, cur_sample_size)
                # print('after body execution, ')
                # show_symbol_tabel_list(symbol_table_list)
            else:
                symbol_table_list = list()
            
            count += 1
            if count > PROTECTION_LOOP_NUM:
                break
        
        # print('======Out Loop======')
        # print(len(res_symbol_table_list))
        # show_symbol_tabel_list(res_symbol_table_list)
        
        return run_next_stmt(self.next_stmt, res_symbol_table_list, cur_sample_size)
        


            


            




        



            

            






        
        