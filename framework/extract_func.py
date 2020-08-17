from initialization import *
from domain import *
from helper import *

import torch
from torch.autograd import Variable
import random
from sympy import Symbol, solve
import seaborn as sns
import matplotlib.pyplot as plt


theta_value = Symbol('theta_value', real=True)
expr_0 = theta_value


def condition_0(Theta):
     
     f = Theta

     return f


def body_0_0(X, Theta):

    C1 = Variable(torch.tensor(-5, dtype=torch.float))
    f = X.add(C1)

    return f


def body_0_1(X, Theta):

    C1 = Variable(torch.tensor(5, dtype=torch.float))
    f = X.add(C1)

    return f


class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.v = val


class SyntaxTree:
    def __init__(self):
        self.root = None


class Condition:
    def __init__(self, branch):
        self.branch = branch
        self.property = 'condition'
    
    def __call__(self, Theta):
        f = globals()['condition_' + str(self.branch)](Theta)
        return
    
    def getProperty(self):
        return self.property


class Body:
    def __init__(self, branch, dirt):
        self.branch = branch
        self.dirt = dirt
        self.property = 'body'
    
    def __call__(self, X, Theta):
        f = globals()['body_' + str(self.branch) + '_' + str(self.dirt)](X, Theta)
        return f

    def getProperty(self):
        return self.property


syntax_tree = SyntaxTree()
syntax_tree.root = Node(Condition(0))
syntax_tree.root.l = Node(Body(0, 0))
syntax_tree.root.r = Node(Body(0, 1))


x_l = 0.0
x_r = 1.0

n_infinity_value = -1000
p_infinity_value = 1000

X_l = Variable(torch.tensor(x_l, dtype=torch.float))
X_r = Variable(torch.tensor(x_r, dtype=torch.float))

n_infinity = Variable(torch.tensor(n_infinity_value, dtype=torch.float))
p_infinity = Variable(torch.tensor(p_infinity_value, dtype=torch.float))

epsilon = Variable(torch.tensor(0.0001, dtype=torch.float))

ini_p_state = program_state_ini


def visit_node(node, program_state):
    if node.v is None:
        return None

    if node.l is None and node.r is None:
        return update_state(program_state, node.v)
    
    l_state = visit_node(node.l, program_state)
    r_state = visit_node(node.r, program_state)

    final_state = magical_join(l_state, r_state)

    return final_state


def magic_join(l_state, r_state):
    # print('l_state, r_state', l_state, r_state)
    if l_state is None:
        return r_state
    if r_state is None:
        return l_state

    C1 = Variable(torch.tensor(1.0, dtype=torch.float))
    C2 = Variable(torch.tensor(0.5, dtype=torch.float))
    C3 = Variable(torch.tensor(2.0, dtype=torch.float))

    c_l = (l_state['abstract_state'][0].add(l_state['abstract_state'][1])).div(C3)
    c_r = (r_state['abstract_state'][0].add(r_state['abstract_state'][1])).div(C3)
    c_out = ((l_state['p'].mul(c_l)).add(r_state['p'].mul(c_r))).div(l_state['p'].add(r_state['p']))
    # print('c_l, c_r, c_out', c_l, c_r, c_out)

    p_l = l_state['p'].div(torch.max(l_state['p'], r_state['p']))
    p_r = r_state['p'].div(torch.max(l_state['p'], r_state['p']))
    # print('p_l, p_r', p_l, p_r)

    c_l_prime = (p_l.mul(c_l)).add((C1.sub(p_l)).mul(c_out))
    c_r_prime = (p_r.mul(c_r)).add((C1.sub(p_r)).mul(c_out))
    # print('c_l_prime, c_r_prime', c_l_prime, c_r_prime)

    l_length = l_state['abstract_state'][1].sub(l_state['abstract_state'][0])
    r_length = r_state['abstract_state'][1].sub(r_state['abstract_state'][0])
    # print('l_length, r_length', l_length, r_length)

    new_l_abstract_state = (c_l_prime.sub(l_length.div(C3).mul(p_l)), c_l_prime.add(l_length.div(C3).mul(p_l)))
    new_r_abstract_state = (c_r_prime.sub(r_length.div(C3).mul(p_r)), c_r_prime.add(r_length.div(C3).mul(p_r)))
    # print(new_l_abstract_state, new_r_abstract_state)

    new_abstract_state = (torch.min(new_l_abstract_state[0], new_r_abstract_state[0]), torch.max(new_l_abstract_state[1], new_r_abstract_state[1]))
    new_w = C1
    new_S = l_state['S'].mul(p_l).add(r_state['S'].mul(p_r))
    new_p = torch.max(C1, p_l.add(p_r))

    new_program_state = dict()
    new_program_state['abstract_state'] = new_abstract_state
    new_program_state['w'] = new_w
    new_program_state['S'] = new_S
    new_program_state['p'] = new_p

    # print('new program state', new_program_state)
    return new_program_state


def cal_func(X_l, X_r, Theta, beta):
    C0 = Variable(torch.tensor(0.0, dtype=torch.float))
    C1 = Variable(torch.tensor(1.0, dtype=torch.float))
    C2 = Variable(torch.tensor(0.5, dtype=torch.float))
    C3 = Variable(torch.tensor(5.0, dtype=torch.float))
    l_C1 = Variable(torch.tensor(1.0, dtype=torch.float))
    r_C1 = Variable(torch.tensor(10.0, dtype=torch.float))
    

    x_l_value = X_l.item()
    x_r_value = X_r.item()
    theta_value = Theta.item()
    # print(x_l_value, x_r_value)

    l_program_state = None
    r_program_state = None

    if x_l_value < theta_value:  # x = x - 5
        l_program_state = dict()
        l_program_state['abstract_state'] = (torch.max(X_l, n_infinity).sub(C3), torch.min(X_r, Theta).sub(C3)) 
        l_program_state['w'] = C1
        l_program_state['S'] = C0.add(l_C1)
        l_program_state['p'] = torch.min(C1, (torch.min(Theta, X_r).sub(X_l)).div(torch.max(epsilon, X_r.sub(X_l)).mul(torch.min(C2, beta))))
        # print('l abstract state', l_program_state['abstract_state'])
        # print('l program state p', l_program_state['p'])
        # print('fen zi', (torch.min(Theta, X_r).sub(X_l)))
        # print('fen mu', torch.max(epsilon, X_r.sub(X_l)).mul(torch.min(C2, beta)))
    
    if theta_value < x_r_value: # x = x + 5
        # print('test')
        r_program_state = dict()
        r_program_state['abstract_state'] = (torch.max(X_l, Theta).add(C3), torch.min(X_r, p_infinity).add(C3))
        r_program_state['w'] = C1
        r_program_state['S'] = C0.add(r_C1)
        r_program_state['p'] = torch.min(C1, (torch.min(X_r, p_infinity).sub(torch.max(X_l, Theta))).div((torch.max(epsilon, X_r.sub(X_l))).mul(torch.min(C2, beta))))
        # print('r abstract state', r_program_state['abstract_state'])
        # print('r program state p', r_program_state['p'])
        # print('fen zi', torch.min(X_r, p_infinity).sub(torch.max(X_l, Theta)))
        # print('fen mu', (torch.max(epsilon, X_r.sub(X_l))).mul(torch.min(C2, beta)))

    final_program_state = magic_join(l_program_state, r_program_state)

    f = final_program_state['abstract_state'][1] # calculate the upper bound

    return f


if __name__ == "__main__":

    beta_value = 1000000
    beta = Variable(torch.tensor(beta_value, dtype=torch.float))
    smooth_x = list()
    smooth_y = list()
    
    for value in range(-500, 1500, 1):
        theta = value * 1.0 / 1000
        Theta = Variable(torch.tensor(theta, dtype=torch.float))
        f = cal_func(X_l, X_r, Theta, beta)
        print('theta, F', theta, f.data.item())
        smooth_x.append(theta)
        smooth_y.append(f.data.item())

    
    beta_value = 0.0000001
    beta = Variable(torch.tensor(beta_value, dtype=torch.float))
    original_x = list()
    original_y = list()
    
    for value in range(-500, 1500, 1):
        theta = value * 1.0 / 1000
        Theta = Variable(torch.tensor(theta, dtype=torch.float))
        f = cal_func(X_l, X_r, Theta, beta)
        print('theta, F', theta, f.data.item())
        original_x.append(theta)                 
        original_y.append(f.data.item())

    plt.plot(smooth_x, smooth_y, label = "beta = 10^6")
    plt.plot(original_x, original_y, label = "beta = 10^(-6)")
    plt.xlabel('theta')
    plt.ylabel('upper bound of f(x)')
    plt.legend()
    plt.show()


    



# return a function with Theta as the parameter

