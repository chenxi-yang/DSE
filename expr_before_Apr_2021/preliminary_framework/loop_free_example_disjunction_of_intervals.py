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
from helper import *
from generate_data import *


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
    partial_intersection_value = intersection.getVolumn().div(X.getVolumn())
    # if partial_intersection_value.data.item() < 0.5:
    #     partial_value = partial_intersection_value.div(f_beta(BETA))
    # else:
    #     partial_value = var(1.0).sub((var(1.0).sub(partial_intersection_value)).div(f_beta(BETA)))

    # partial_value = intersection.getVolumn().div(X.getVolumn().mul(f_beta(BETA)))
    # print('partial_value', partial_value)

    return torch.min(var(1.0), partial_intersection_value)


def update_symbol_table_with_constraint(target, test, symbol_table, direct):
    constraint_interval = domain.Interval(-10000.0, 10000.0)
    res_symbol_table = dict()
    # self-deepcopy symbol_table
    for symbol in symbol_table:
        if symbol == 'probability':
            res_symbol_table[symbol] = var(symbol_table[symbol].data.item())
        else:
            res_symbol_table[symbol] = domain.Interval(symbol_table[symbol].left.data.item(), symbol_table[symbol].right.data.item())
    
    target_value = res_symbol_table[target]
    # print('------in constraint')
    # print('target-name', target)
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
        if intersection_interval.left.data.item() == intersection_interval.right.data.item() and (intersection_interval.left.data.item() == constraint_interval.left.data.item() or intersection_interval.right.data.item() == constraint_interval.right.data.item()):
            intersection_interval = None
            probability = var(0.0)
        else:
            # print('beta', f_beta(BETA))
            # print('pho', pho(target_value, intersection_interval))
            probability = symbol_table['probability'].mul(pho(target_value, intersection_interval))

    res_symbol_table[target] = intersection_interval
    res_symbol_table['probability'] = probability
    # print('probability', res_symbol_table['probability'])
    # print('final intersection', intersection_interval)
    # print('------out constraint')

    return [res_symbol_table]


def cal_branch_weight(target, test, symbol_table):
    # print('target, test', symbol_table[target].data.item(), test.data.item())
    
    diff = symbol_table[target].sub(test)
    alpha = var(0.5).mul(var(1.0).add(torch.tanh(f_beta(BETA).mul(diff))))

    w_body = var(1.0).sub(alpha)
    w_orelse = alpha

    # print('w1, w2', w_body.data.item(), w_orelse.data.item())

    return w_body, w_orelse


def update_symbol_table(target, func, symbol_table):
    #! assume only monotone function
    target_value = symbol_table[target]

    symbol_table[target].left = func(target_value.left)
    symbol_table[target].right = func(target_value.right)

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

    return symbol_table


class Ifelse:
    def __init__(self, target, test, body, orelse, next_stmt):
        self.target = target
        self.test = test
        self.body = body
        self.orelse = orelse
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table_list):
        # print('--- in ifelse target', self.target)
        # show(symbol_table)
        res_symbol_table_list = list()

        for symbol_table in symbol_table_list:
            this_block_p = symbol_table['probability'].data.item()
            body_symbol_table_list = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '<')
            orelse_symbol_table_list = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '>')
            # print('after if else split', self.target)
            # print('--body')
            # show(body_symbol_table)
            # print('--orelse')
            # show(orelse_symbol_table)

            if body_symbol_table_list[0]['probability'].data.item() > 0.0:
                body_symbol_table_list = self.body.execute(body_symbol_table_list)
                res_symbol_table_list.extend(body_symbol_table_list)
            # else:
            #     return run_next_stmt(self.next_stmt, orelse_symbol_table)
            
            # print('a show1')
            # print('--1--body')
            # show(body_symbol_table)
            # print('--1--orelse')
            # show(orelse_symbol_table)
            # print('b show1')
            
            if orelse_symbol_table_list[0]['probability'].data.item() > 0.0:
                orelse_symbol_table_list = self.orelse.execute(orelse_symbol_table_list)
                res_symbol_table_list.extend(orelse_symbol_table_list)
            # else:
            #     return run_next_stmt(self.next_stmt, body_symbol_table)

            # print('a show2')
            # print('--2--body')
            # show(body_symbol_table)
            # print('--2--orelse')
            # show(orelse_symbol_table)
            # print('b show2')

            # symbol_table = join(body_symbol_table, orelse_symbol_table)
            # symbol_table['probability'] = var(this_block_p)

            # print('--after join')
            # show(symbol_table)

            # print('---end if')

        return run_next_stmt(self.next_stmt, res_symbol_table_list)


class WhileSimple:
    #! not a real loop, just in the form of loop to operate several if-else stmt
    def __init__(self, target, test, body, next_stmt):
        # TODO: implement while & test
        self.target = target
        self.test = test
        self.body = body
        self.next_stmt = next_stmt
     
    def execute(self, symbol_table_list):

        this_block_p = symbol_table_list[0]['probability'].data.item()
        
        body_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table_list[0], '<')
        # orelse_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '>')

        i_list = list()
        l_list = list() 
        r_list = list()

        l_list = [100000 for i in range(41)]
        r_list = [-100000 for i in range(41)]

        l_list[0] = symbol_table_list[0]['x'].left.data.item()
        r_list[0] = symbol_table_list[0]['x'].right.data.item()
        i_list = [i for i in range(41)]

        # i_list.append(symbol_table['i'].left.data.item())
        # l_list.append(symbol_table['x'].left.data.item())
        # r_list.append(symbol_table['x'].right.data.item())
        # print('11111, pro', body_symbol_table['probability'].data.item(), (var(1.0).div(f_beta(BETA))).data.item())

        #! if body_symbol_table['probability'].data.item() == (var(1.0).div(f_beta(BETA))).data.item():
        #     print('111-----------------------------this block p', this_block_p)
        body_symbol_table[0]['probability'] = var(1.0)

        symbol_table_queue = queue.Queue()
        symbol_table_queue.put(body_symbol_table[0])

        res_symbol_table_list = list()

        while(not symbol_table_queue.empty()):
            body_symbol_table = symbol_table_queue.get()

            if body_symbol_table['probability'].data.item() > 0:
            
                body_symbol_table_list = self.body.execute([body_symbol_table])

                # show_symbol_tabel_list(body_symbol_table_list)
                for tmp_body_symbol_table in body_symbol_table_list:
                    idx = int(tmp_body_symbol_table['i'].left.data.item())
                    l_value = tmp_body_symbol_table['x'].left.data.item()
                    r_value = tmp_body_symbol_table['x'].right.data.item()

                    l_list[idx] = min(l_list[idx], l_value)
                    r_list[idx] = max(r_list[idx], r_value)

                flag = 0.0

                for tmp_body_symbol_table in body_symbol_table_list:
                    new_tmp_body_symbol_table_list = update_symbol_table_with_constraint(self.target, self.test, tmp_body_symbol_table, '<')
                    if new_tmp_body_symbol_table_list[0]['probability'].data.item() <= 0.0:
                        flag = 1.0
                        break
                    symbol_table_queue.put(new_tmp_body_symbol_table_list[0])
                
                if flag == 1.0:
                    for tmp_body_symbol_table in body_symbol_table_list:
                        res_symbol_table_list.append(tmp_body_symbol_table)
            
        for i, value in enumerate(l_list):
            print(l_list[i], r_list[i])
        plt.plot(i_list, l_list, label="left", marker='x')
        plt.plot(i_list, r_list, label="right", marker ='x')
        plt.xlabel('loop')
        plt.ylabel('l-r')
        plt.legend()
        plt.show()

        return run_next_stmt(self.next_stmt, res_symbol_table_list)


class Assign:
    def __init__(self, target, value, next_stmt):
        self.target = target
        self.value = value
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table_list):
        for idx, symbol_table in enumerate(symbol_table_list):
            symbol_table_list[idx] = update_symbol_table(self.target, self.value, symbol_table)

        return run_next_stmt(self.next_stmt, symbol_table_list)


def update_symbol_table_point(target, func, symbol_table):

    # show_symbol_table_point(symbol_table, '352')

    res_symbol_table = dict()
    for symbol in symbol_table:
        if symbol == target:
            res_symbol_table[symbol] = func(symbol_table[symbol])# func(var(symbol_table[target].data.item()))
        else:
            res_symbol_table[symbol] = symbol_table[symbol] # var(symbol_table[symbol].data.item()) 
    
    # show_symbol_table_point(res_symbol_table, '361')

    return res_symbol_table


def smooth_branch(symbol_table_body, symbol_table_orelse, w_body, w_orelse):

    # show_symbol_table_point(symbol_table_body, 'body-368')
    # show_symbol_table_point(symbol_table_orelse, 'orelse-369')

    symbol_table = dict()
    for symbol in symbol_table_body:
        if symbol_table_body[symbol].data.item() == symbol_table_orelse[symbol].data.item():
            symbol_table[symbol] = symbol_table_body[symbol]
        else:
            symbol_table[symbol] = (w_body.mul(symbol_table_body[symbol])).add(w_orelse.mul(symbol_table_orelse[symbol]))

    # show_symbol_table_point(symbol_table, 'all-378')
    return symbol_table


def show_symbol_table_point(symbol_table, loc):

    print('==============' + loc + '===============')
    for symbol in symbol_table:
        print(symbol, symbol_table[symbol])
    print('=============================')

    return 


class AssignPoint:
    def __init__(self, target, value, next_stmt):
        self.target = target
        self.value = value
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):
        symbol_table = update_symbol_table_point(self.target, self.value, symbol_table)

        return run_next_stmt(self.next_stmt, symbol_table)


class IfelsePoint:
    def __init__(self, target, test, body, orelse, next_stmt):
        self.target = target
        self.test = test
        self.body = body
        self.orelse = orelse
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):
        if symbol_table[self.target].data.item() < self.test.data.item():
            symbol_table = self.body.execute(symbol_table)
        else:
            symbol_table = self.orelse.execute(symbol_table)
        
        return run_next_stmt(self.next_stmt, symbol_table)


class WhileSimplePoint:
    #! not a real loop, just in the form of loop to operate several if-else stmt
    def __init__(self, target, test, body, next_stmt):
        # TODO: implement while & test
        self.target = target
        self.test = test
        self.body = body
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):
        while(symbol_table[self.target].data.item() < self.test.data.item()):
            symbol_table = self.body.execute(symbol_table)
            # print('WhileSimplePoint: i, x', symbol_table['i'], symbol_table['x'])
        
        return run_next_stmt(self.next_stmt, symbol_table)
# TODO: smooth & initialization


class AssignPointSmooth:
    def __init__(self, target, value, next_stmt):
        self.target = target
        self.value = value
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):
        symbol_table = update_symbol_table_point(self.target, self.value, symbol_table)

        return run_next_stmt(self.next_stmt, symbol_table)


class IfelsePointSmooth:
    def __init__(self, target, test, body, orelse, next_stmt):
        self.target = target
        self.test = test
        self.body = body
        self.orelse = orelse
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):

        w_body, w_orelse = cal_branch_weight(self.target, self.test, symbol_table)

        symbol_table_body = self.body.execute(symbol_table)
        symbol_table_orelse = self.orelse.execute(symbol_table)

        symbol_table = smooth_branch(symbol_table_body, symbol_table_orelse, w_body, w_orelse)
        
        return run_next_stmt(self.next_stmt, symbol_table)


class WhileSimplePointSmooth:
    #! not a real loop, just in the form of loop to operate several if-else stmt
    def __init__(self, target, test, body, next_stmt):
        # TODO: implement while & test
        self.target = target
        self.test = test
        self.body = body
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):
        while(symbol_table[self.target].data.item() < self.test.data.item()):
            # print('WhileSimplePointSmooth: i, x', symbol_table['i'], symbol_table['x'])
            symbol_table = self.body.execute(symbol_table)
        
        return run_next_stmt(self.next_stmt, symbol_table)
# TODO: smooth & initialization


# def initialization(x_l, x_r):

#     symbol_table = dict()
#     symbol_table['i'] = domain.Interval(0, 0)
#     symbol_table['x'] = domain.Interval(x_l, x_r)
#     symbol_table['isOn'] = domain.Interval(0.0, 0.0)
#     symbol_table['probability'] = var(1.0)
    
#     return symbol_table


def initialization_point(x):
    symbol_table_point = dict()
    symbol_table_point['i'] = var(0.0)
    symbol_table_point['x'] = var(x)
    symbol_table_point['isOn'] = var(0.0)

    return symbol_table_point


def initialization(x_l, x_r):

    symbol_table_list = list()

    symbol_table = dict()
    symbol_table['i'] = domain.Interval(0, 0)
    symbol_table['x'] = domain.Interval(x_l, x_r)
    symbol_table['isOn'] = domain.Interval(0.0, 0.0)
    symbol_table['probability'] = var(1.0)

    symbol_table_list.append(symbol_table)
    
    return symbol_table_list


def f6(x):
    return x.sub(var(0.1).mul(x.sub(var(60))))
def f18(x):
    return x.add(var(1.0))
def f8(x):
    return var(1.0)
def f10(x):
    return var(0.0)
def f12(x):
    return x.sub(var(0.1).mul(x.sub(var(60)))).add(var(5.0))


def construct_syntax_tree(Theta):
    
    l18 = Assign('i', f18, None)

    l8 = Assign('isOn', f8, None)
    l10 = Assign('isOn', f10, None)
    l7 = Ifelse('x', Theta, l8, l10, None)

    l6 = Assign('x', f6, l7)

    l14 = Assign('isOn', f8, None)
    l16 = Assign('isOn', f10, None)
    l13 = Ifelse('x', var(80.0), l14, l16, None)

    l12 = Assign('x', f12, l13)
    l5 = Ifelse('isOn', var(0.5), l6, l12, l18)

    l4 = WhileSimple('i', var(40.0), l5, None)

    return l4


def construct_syntax_tree_point(Theta):

    l18 = AssignPoint('i', f18, None)

    l8 = AssignPoint('isOn', f8, None)
    l10 = AssignPoint('isOn', f10, None)
    l7 = IfelsePoint('x', Theta, l8, l10, None)

    l6 = AssignPoint('x', f6, l7)

    l14 = AssignPoint('isOn', f8, None)
    l16 = AssignPoint('isOn', f10, None)
    l13 = IfelsePoint('x', var(80.0), l14, l16, None)

    l12 = AssignPoint('x', f12, l13)
    l5 = IfelsePoint('isOn', var(0.5), l6, l12, l18)

    l4 = WhileSimplePoint('i', var(40), l5, None)

    return l4


def construct_syntax_tree_smooth_point(Theta):

    l18 = AssignPointSmooth('i', f18, None)

    l8 = AssignPointSmooth('isOn', f8, None)
    l10 = AssignPointSmooth('isOn', f10, None)
    l7 = IfelsePointSmooth('x', Theta, l8, l10, None)

    l6 = AssignPointSmooth('x', f6, l7)

    l14 = AssignPointSmooth('isOn', f8, None)
    l16 = AssignPointSmooth('isOn', f10, None)
    l13 = IfelsePointSmooth('x', var(80.0), l14, l16, None)

    l12 = AssignPointSmooth('x', f12, l13)
    l5 = IfelsePointSmooth('isOn', var(0.5), l6, l12, l18)

    l4 = WhileSimplePointSmooth('i', var(40.0), l5, None)

    return l4


# def distance_f_point(X, target):
#     X_length = X.getLength()
#     if target.data.item() < X.right.data.item() and target.data.item() > X.left.data.item():
#         res = C0
#     else:
#         res = torch.max(target.sub(X.right), X.left.sub(target)).div(X_length)
    
#     return res

def distance_f_point(pred_y, y):
    return torch.abs(pred_y.sub(y))


# def distance_f_point(X, target):
#     X_length = X.getLength()
#     if target.data.item() < X.right.data.item() and target.data.item() > X.left.data.item():
#         res = C0
#     else:
#         res = torch.max(target.sub(X.right), X.left.sub(target)).div(X_length)
    
#     return res


def distance_f_interval(X_list, target):

    res = var(0.0)
    for X_table in X_list:
        X = X_table['x']
        p = X_table['probability']

        tmp_res = var(0.0)
        intersection_interval = get_intersection(X, target)
        # print('intersection:', intersection_interval.left, intersection_interval.right)
        if intersection_interval.isEmpty():
            # print('isempty')
            tmp_res = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            # print('not empty')
            tmp_res = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
        # print(X.left, X.right, tmp_res)
        tmp_res = tmp_res.mul(p)

        res = res.add(tmp_res)
    
    return res


def show(symbol_table):
    print('symbol table:')
    for symbol in symbol_table:
        if symbol == 'probability':
            print(symbol, symbol_table[symbol])
        else:
            if symbol_table[symbol] is None:
                print(symbol, 'None')
            else:
                print(symbol, symbol_table[symbol].left, symbol_table[symbol].right)
    
    return 


def show_symbol_tabel_list(symbol_table_list):
    print('symbol table list:')
    # l_min = 100000
    # r_max = -10000
    for symbol_table in symbol_table_list:
        p = symbol_table['probability'].data.item()
        l = symbol_table['x'].left.data.item()
        r = symbol_table['x'].right.data.item()
        print('probability: ' + str(p) + ', interval: ' + str(l) + ',' + str(r))

        # l_min = min(l, l_min)
        # r_max = max(r, r_max)
    # print('disjunction falls in', l_min, r_max)


def plot_func(root, Theta, ltarget, x_l, x_r):
    original_x = list()
    original_y = list()

    left_x = list()
    left_y = list()    
    right_x = list()
    right_y = list()

    for i in range(64, 76, 1):
        x = i * 1.0 / 1
        x = 70.0

        Theta.data = var(x)

        symbol_table_list = initialization(x_l, x_r)
        symbol_table_list = root.execute(symbol_table_list)

        show_symbol_tabel_list(symbol_table_list)
        print('final res length', len(symbol_table_list))

        f = distance_f_interval(symbol_table_list, target)

        # f_r = symbol_table['x'].right
        # f_l = symbol_table['x'].left
        print('theta, x', x, f.data.item())

        # left_x.append(x)
        # left_y.append(f_l.data.item())
        # right_x.append(x)
        # right_y.append(f_r.data.item())

        break
        exit(0)

        original_x.append(x)
        original_y.append(f.data.item())
        # break
    
    # plt.plot(smooth_x, smooth_y, label = "beta = 10^6")
    # plt.plot(original_x, original_y, label = "beta")
    # plt.xlabel('theta')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.show()

    # plt.plot(left_x, left_y, label="left", marker='x')
    # plt.plot(right_x, right_y, label="right", marker='x')
    # plt.xlabel('theta')
    # plt.ylabel('Interval')
    # plt.legend()
    # plt.show()


def test_point_func(X_df, root_smooth_point):
    for idx, row in X_df.iterrows():
        x, y = row['x'], row['y']
        symbol_table_point = initialization_point(x)
        symbol_table_point = root_smooth_point.execute(symbol_table_point)
        
        print('x, diff(pred_y, y)', x, symbol_table_point['x'].data.item(), y)

        break


def data_generator(l, r, root, num):
    data_list = list()

    for i in range(num):
        x = random.uniform(l, r)

        symbol_table_point = initialization_point(x)
        symbol_table_point = root.execute(symbol_table_point)
        y = symbol_table_point['x'].data.item()

        data = [x, y]
        data_list.append(data)
    
    return pd.DataFrame(data_list, columns=['x', 'y'])


def plot_quantative_func(root_smooth_point, Theta, X_df):
    original_x = list()
    original_y = list()

    length = X_df.shape[0]
    print(length)
    
    for i in range(60, 80, 2):
        theta = i * 1.0 / 1
        Theta.data = var(theta)

        f = var(0.0)

        f_x = list()
        f_pred_y = list()
        f_y= list()

        for idx, row in X_df.iterrows():
            x, y = row['x'], row['y']
            symbol_table_point = initialization_point(x)
            symbol_table_point = root_smooth_point.execute(symbol_table_point)

            f_x.append(x)
            f_pred_y.append(symbol_table_point['x'].data.item())
            f_y.append(y)

            print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
            f = f.add(distance_f_point(symbol_table_point['x'], var(y)))
            # f = symbol_table_point['x']
        
        # f = f.div(var(length))

        print('theta, f', theta, f.data.item())
        plt.scatter(f_x, f_pred_y, label='pred')
        plt.scatter(f_x, f_y, label='y')
        plt.xlabel('x')
        plt.ylabel('f(x, theta)')
        plt.legend()
        plt.show()

        original_x.append(theta)
        original_y.append(f.data.item())

    plt.plot(original_x, original_y, label = "beta")
    plt.xlabel('theta')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
    


'''
# Meta-test
class Test:
    def __init__(self, value):
        self.value = value
'''

if __name__ == "__main__":

    lr = 0.2
    epoch = 1000
    dataset_size = 50
    Theta_ini = var(70.0)
    Theta = var(random.uniform(55.0, 76.0), requires_grad=True)
    # target = var(75.0, requires_grad=True)
    target = domain.Interval(68.0, 79.0)
    lambda_ = 2.0
    x_l = 65.0
    x_r= 75.0 

    root_point = construct_syntax_tree_point(Theta_ini)

    X_df = data_generator(x_l, x_r, root_point, dataset_size) # theta is the value of synthesized objective

    symbol_table_point = initialization_point(random.uniform(x_l, x_r))
    root = construct_syntax_tree(Theta)
    root_smooth_point = construct_syntax_tree_smooth_point(Theta)

    plot_func(root, Theta, target, x_l, x_r)

    # # DIRECT
    # loop_list = list()
    # loss_list = list()
    # loop_list.append(1)
    # # global count = 0
    # # count = 0

    # def myfunc(theta, grad):
    #     global count
    #     # print(x)
    #     Theta = var(theta)
    #     # Theta = var(70.0)
    #     root = construct_syntax_tree(Theta)
    #     symbol_table_list = initialization(x_l, x_r)
    #     # root_point = construct_syntax_tree_point(Theta)
    #     root_smooth_point = construct_syntax_tree_smooth_point(Theta)

    #     f = var(0.0)

    #     for idx, row in X_df.iterrows():
    #         x, y = row['x'], row['y']
    #         symbol_table_point = initialization_point(x)
    #         symbol_table_point = root_smooth_point.execute(symbol_table_point)

    #         # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
    #         f = f.add(distance_f_point(symbol_table_point['x'], var(y)))
    #     f = f.div(var(dataset_size))
    #     print('quantitive f', f.data.item())

    #     symbol_table_list = root.execute(symbol_table_list)
    #     # show_symbol_tabel_list(symbol_table_list)
    #     # print(len(symbol_table_list))
    #     penalty_f = distance_f_interval(symbol_table_list, target)
    #     print('safe f', penalty_f.data.item())

    #     f = f.add(var(lambda_).mul(penalty_f))
    #     print(Theta.data.item(), f.data.item())
    #     f_value = f.data.item()

    #     count = loop_list[-1]
    #     loop_list.append(count + 1)
    #     loss_list.append(f_value)

    #     if abs(f_value) < epsilon_value:
    #         raise ValueError(str(theta[0]) + ',' + str(f_value))

    #     # exit(0)

    #     return f_value

    # x = np.array([66.0])
    # opt = nlopt.opt(nlopt.GN_DIRECT, 1)
    # opt.set_lower_bounds([55.0])
    # opt.set_upper_bounds([78.0])
    # opt.set_min_objective(myfunc)
    # opt.set_stopval(0.5)
    # opt.set_maxeval(1000)
    # try:
    #     x = opt.optimize(x)
    # except ValueError as error:
    #     error_list = str(error).split(',')
    #     error_value = [float(err) for err in error_list]
    #     print('theta, f', error_value[0], error_value[1])
    
    # plt.plot(loop_list[:-1], loss_list, label = "beta")
    # plt.xlabel('expr count')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()


    # GD + noise
    # derivation = var(0.0)
    loop_list = list()
    loss_list = list()
    for i in range(epoch):

        symbol_table_list = initialization(x_l, x_r)

        f = var(0.0)

        for idx, row in X_df.iterrows():
            x, y = row['x'], row['y']
            symbol_table_point = initialization_point(x)
            symbol_table_point = root_smooth_point.execute(symbol_table_point)

            # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
            f = f.add(distance_f_point(symbol_table_point['x'], var(y)))
        f = f.div(var(dataset_size))
        print('quantitive f', f.data.item())

        symbol_table_list = root.execute(symbol_table_list)
        penalty_f = distance_f_interval(symbol_table_list, target)

        # print('safe f', penalty_f.data.item())

        f = f.add(var(lambda_).mul(penalty_f))
        # print(Theta.data.item(), f.data.item())

        try:
            dTheta = torch.autograd.grad(f, Theta, retain_graph=True)
            derivation = dTheta[0]
            print('f, theta, dTheta:', f.data, Theta.data, derivation)

            if torch.abs(f.data) < var(0.5): # epsilon:
                print(f.data, Theta.data)
                break
        
            Theta.data += lr * (derivation.data + var(random.uniform(-20.0, 20.0)))
        
        except RuntimeError:
            print('RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn')
            if torch.abs(f.data) < epsilon:
                print(f.data, Theta.data)
                break

            Theta.data += lr * (derivation.data + var(random.uniform(-20.0, 20.0)))
        
        if Theta.data.item() <= 55.0 or Theta.data.item() >= 76.0:
            Theta.data.fill_(random.uniform(55.0, 76.0))
            continue

        loop_list.append(i)
        loss_list.append(f.data)
    
    plt.plot(loop_list, loss_list, label = "beta")
    plt.xlabel('expr count')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    print('GOT! Loss, theta', f.data, Theta.data)
    # show(symbol_table)


    '''
    # Meta-test
    a = var(1.0)

    l = Test(a)

    a.data += a.data

    print(l.value)
    '''
    


