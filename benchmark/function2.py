import torch
from torch.autograd import Variable
import random
from sympy import Symbol, solve

'''
# program
if(x < theta):
    if(x < theta/2):
        x = -x
    else:
        x = -x + 1
else:
    if(x < theta + 12):
        x = x + theta
    else:
        x = x
'''


theta_value = Symbol('theta_value', real=True)
expr_0 = theta_value
expr_1 = theta_value / 2
expr_2 = theta_value + 12

n_infinity_value = -1000
p_infinity_value = 1000
x_l = -5
x_r = 5
c_l = -11
c_r = -1


def condition_0(Theta):

    f = Theta

    return f


def body_0_0(X, Theta):

    f = X

    return f


def condition_1(Theta):

    C1 = Variable(torch.tensor(2, dtype=torch.float))

    f = Theta.div(C1)

    return f


def body_1_0(X, Theta):

    C1 = Variable(torch.tensor(-1, dtype=torch.float))

    f = torch.mul(C1, X)

    return f


def body_1_1(X, Theta):

    C1 = Variable(torch.tensor(-1, dtype=torch.float))
    C2 = Variable(torch.tensor(1, dtype=torch.float))

    f = torch.mul(C1, X).add(C2)

    return f


def condition_2(Theta):

    C1 = Variable(torch.tensor(12, dtype=torch.float))

    f = Theta.add(C1)

    return f


def body_2_0(X, Theta):

    # C1 = Variable(torch.tensor(1, dtype=torch.float))

    f = X.add(Theta)

    return f


def body_2_1(X, Theta):

    f = X

    return f


class Condition:

    def __init__(self, branch):
        self.branch = branch
    
    def __call__(self, Theta):
        f = globals()['condition_' + str(self.branch)](Theta)
        return f


class Body:

    def __init__(self, branch, dirt): # dirt 0: >, dirt 1: <
        self.branch = branch
        self.dirt = dirt
    
    def __call__(self, X, Theta):
        f = globals()['body_' + str(self.branch) + '_' + str(self.dirt)](X, Theta)
        return f


class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.v = val

class Tree:
    def __init__(self):
        self.root = None

    def getRoot(self):
        return self.root

    def add(self, val):
        if(self.root == None):
            self.root = Node(val)
        else:
            self._add(val, self.root)

    def _add(self, val, node):
        if(val < node.v):
            if(node.l != None):
                self._add(val, node.l)
            else:
                node.l = Node(val)
        else:
            if(node.r != None):
                self._add(val, node.r)
            else:
                node.r = Node(val)

    def printTree(self):
        if(self.root != None):
            self._printTree(self.root)

    def _printTree(self, node):
        if(node != None):
            self._printTree(node.l)
            # print str(node.v) + ' '
            self._printTree(node.r)
