import torch
import random
from torch.autograd import Variable
from numpy import *

from constants import *
from helper import *


def f_beta_smooth_point(beta):
    gamma = var(0.1)
    # return torch.min(var(0.5), beta)
    return beta


def run_next_stmt(next_stmt, symbol_table):
    if next_stmt is None:
        return symbol_table
    else:
        return next_stmt.execute(symbol_table)

def show_tmp_x_y(symbol_table):
    print('symbol_table:')
    print(symbol_table['x1'], symbol_table['y1'], symbol_table['x2'], symbol_table['y2'])

# def update_symbol_table_point(target, func, symbol_table):

#     # show_symbol_table_point(symbol_table, '352')

#     res_symbol_table = dict()
#     for symbol in symbol_table:
#         if symbol == target:
#             res_symbol_table[symbol] = func(symbol_table[symbol])# func(var(symbol_table[target].data.item()))
#         else:
#             res_symbol_table[symbol] = symbol_table[symbol] # var(symbol_table[symbol].data.item()) 
    
#     # show_symbol_table_point(res_symbol_table, '361')

#     return res_symbol_table


# For multiple inputs
def update_symbol_table_point(target, func, symbol_table):

    # show_symbol_table_point(symbol_table, '352')
    # print('line42-', symbol_table['i'])
    res_target = target[0]
    value_list = [symbol_table[symbol] for symbol in target]
    # print(value_list)

    res_symbol_table = dict()
    for symbol in symbol_table:
        if symbol == res_target:
            # print('func')
            # print(value_list)
            # print(func(value_list))
            # print('end')
            res_symbol_table[symbol] = func(value_list)# func(var(symbol_table[target].data.item()))
        else:
            res_symbol_table[symbol] = symbol_table[symbol] # var(symbol_table[symbol].data.item()) 
    
    # show_symbol_table_point(res_symbol_table, '361')
    # print('line55-', symbol_table['i'])
    # print(res_symbol_table[res_target])

    return res_symbol_table


class AssignPoint:
    def __init__(self, target, value, next_stmt):
        self.target = target
        self.value = value
        self.next_stmt = next_stmt
    
    def execute(self, symbol_table):
        # print('---Assign', self.target)
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
        # print('if-else target', self.target)
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
            # print('WhileSimplePoint: i, x, stage', symbol_table['i'], symbol_table['x'], symbol_table['stage'])
            symbol_table = self.body.execute(symbol_table)
            # print('WhileSimplePoint: i, x, stage', symbol_table['i'], symbol_table['x'], symbol_table['stage'])
            # show_tmp_x_y(symbol_table)
            # break
        
        return run_next_stmt(self.next_stmt, symbol_table)


'''
tanh smooth of if-else branch
e = (1-\alpha)e1 + \alpha e2
\alpha = 0.5*(1 + tanh((e - test)*beta))
'''
def cal_branch_weight(target, test, symbol_table):
    # print('target, test', symbol_table[target].data.item(), test.data.item())
    
    diff = symbol_table[target].sub(test)
    alpha = var(0.5).mul(var(1.0).add(torch.tanh(f_beta_smooth_point(POINT_BETA).mul(diff))))

    w_body = var(1.0).sub(alpha)
    w_orelse = alpha

    # print('w1, w2', w_body.data.item(), w_orelse.data.item())

    return w_body, w_orelse


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