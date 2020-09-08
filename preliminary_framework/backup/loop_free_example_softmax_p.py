import torch
import random
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt
import sys
import nlopt
import numpy as np
from numpy import *

import domain
from helper import *


def var(i, requires_grad=False):
    return Variable(torch.tensor(i, dtype=torch.float), requires_grad=requires_grad)

N_INFINITY = var(-10000.0)
P_INFINITY = var(10000.0)

BETA = var(99999.99)

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

    return torch.min(var(2), beta)
    # return beta


def pho(X, intersection):
    x = intersection.getVolumn().div(X.getVolumn())
    partial_value = intersection.getVolumn().div(X.getVolumn()).pow(f_beta(BETA))
    # partial_value = var(1.0).div(var(1.0).add(torch.exp(var(0.0).sub(x.mul(var(1.0)).sub(var(0.5))))))
    
    # if partial_intersection_value.data.item() < 0.5:
    #     partial_value = partial_intersection_value.div(f_beta(BETA))
    # else:
    #     partial_value = var(1.0).sub((var(1.0).sub(partial_intersection_value)).div(f_beta(BETA)))

    # partial_value = intersection.getVolumn().div(X.getVolumn().mul(f_beta(BETA)))
    # print('x', x)
    # print('partial_value', partial_value)

    return torch.min(var(1.0), partial_value)


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

    return res_symbol_table


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
    
    def execute(self, symbol_table):
        # print('--- in ifelse target', self.target)
        # show(symbol_table)

        this_block_p = symbol_table['probability'].data.item()
        body_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '<')
        orelse_symbol_table = update_symbol_table_with_constraint(self.target, self.test, symbol_table, '>')
        # print('after if else split', self.target)
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


def initialization():

    symbol_table = dict()
    symbol_table['i'] = domain.Interval(0, 0)
    symbol_table['x'] = domain.Interval(65.0, 75.0)
    symbol_table['isOn'] = domain.Interval(0.0, 0.0)
    symbol_table['probability'] = var(1.0)
    
    return symbol_table


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

# def f6(x):
#     return x.sub(var(0.1))
# def f18(x):
#     return x.add(var(1.0))
# def f8(x):
#     return var(1.0)
# def f10(x):
#     return var(0.0)
# def f12(x):
#     return x.add(var(5.0))


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


def distance_f_point(X, target):
    X_length = X.getLength()
    if target.data.item() < X.right.data.item() and target.data.item() > X.left.data.item():
        res = C0
    else:
        res = torch.max(target.sub(X.right), X.left.sub(target)).div(X_length)
    
    return res


def distance_f_interval(X, target):
    intersection_interval = get_intersection(X, target)
    if intersection_interval.isEmpty():
        res = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
    else:
        res = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
    
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


def plot_func(root, Theta, ltarget):
    original_x = list()
    original_y = list()

    left_x = list()
    left_y = list()    
    right_x = list()
    right_y = list()

    for i in range(64, 75, 1):
        x = i * 1.0 / 1

        Theta.data = var(x)

        symbol_table = initialization()
        symbol_table = root.execute(symbol_table)
        f_r = symbol_table['x'].right
        f_l = symbol_table['x'].left
        f = distance_f_interval(symbol_table['x'], target)
        print('theta, left, right', x, f_l.data.item(), f_r.data.item())

        left_x.append(x)
        left_y.append(f_l.data.item())
        right_x.append(x)
        right_y.append(f_r.data.item())

        # break

        original_x.append(x)
        original_y.append(f.data.item())
    
    # plt.plot(smooth_x, smooth_y, label = "beta = 10^6")
    plt.plot(original_x, original_y, label = "beta")
    plt.xlabel('theta')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    plt.plot(left_x, left_y, label="left", marker='x')
    plt.plot(right_x, right_y, label="right", marker='x')
    plt.xlabel('theta')
    plt.ylabel('Interval')
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
    Theta = var(68.0, requires_grad=True)
    # target = var(75.0, requires_grad=True)
    target = domain.Interval(68.0, 76.0)

    root = construct_syntax_tree(Theta)

    plot_func(root, Theta, target)

    def myfunc(x, grad):
        # print(x)
        Theta = var(x)
        root = construct_syntax_tree(Theta)
        symbol_table = initialization()
        symbol_table = root.execute(symbol_table)
        # print(len(symbol_table_list))

        f = distance_f_interval(symbol_table['x'], target)
        print(Theta.data.item(), f.data.item())
        f_value = f.data.item()

        if abs(f_value) < epsilon_value:
            raise ValueError(str(x[0]) + ',' + str(f_value))

        return f_value

    x = np.array([66.0])
    opt = nlopt.opt(nlopt.GN_DIRECT, 1)
    opt.set_lower_bounds([55.0])
    opt.set_upper_bounds([78.0])
    opt.set_min_objective(myfunc)
    opt.set_stopval(0.0)
    opt.set_maxeval(1000)
    try:
        x = opt.optimize(x)
    except ValueError as error:
        error_list = str(error).split(',')
        error_value = [float(err) for err in error_list]
        print('theta, f', error_value[0], error_value[1])

    # def myfunc(x, grad):
    #     Theta = var(x[0], requires_grad=True)
        


    # for i in range(epoch):

    #     symbol_table = initialization()
    #     symbol_table = root.execute(symbol_table)
    #     # print('x', symbol_table['x'].left, symbol_table['x'].right)
    #     f = distance_f_point(symbol_table['x'], target)

    #     dTheta = torch.autograd.grad(f, Theta, retain_graph=True)
    #     derivation = dTheta[0]
    #     print('f, theta, dTheta:', f.data, Theta.data, derivation)

    #     if torch.abs(derivation) < epsilon:
    #         print(f.data, Theta.data)
        
    #     Theta.data += lr * derivation.data
    
    # print('Loss, theta', f.data, Theta.data)
    # show(symbol_table)


    '''
    # Meta-test
    a = var(1.0)

    l = Test(a)

    a.data += a.data

    print(l.value)
    '''
    


