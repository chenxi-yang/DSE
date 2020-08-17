import domain
from helper import *
from initialization import *
from function1 import *

import torch
import random
from torch.autograd import Variable
import copy


# assume all the test is a < b

n_infinity_value = -1000.0
p_infinity_value = 1000.0
n_infinity = Variable(torch.tensor(n_infinity_value, dtype=torch.float))
p_infinity = Variable(torch.tensor(p_infinity_value))

theta_on = 66.0
theta_off = 81.0

C_on = Variable(torch.tensor(1.0, dtype=torch.float))
C_off = Variable(torch.tensor(0.0, dtype=torch.float))

constraint_r = Variable(torch.tensor(120.0, dtype=torch.float))
constraint_l = Variable(torch.tensor(p_infinity_value, dtype=torch.float))

C0 = Variable(torch.tensor(0.0, dtype=torch.float))
C1 = Variable(torch.tensor(4, dtype=torch.float))
C2 = Variable(torch.tensor(0.5, dtype=torch.float))
C3 = Variable(torch.tensor(1.0, dtype=torch.float))
C_constraint = Variable(torch.tensor(120.0, dtype=torch.float))

Theta = Variable(torch.tensor([theta_on, theta_off], dtype=torch.float), requires_grad=True)

beta_value = 100000
beta = Variable(torch.tensor(beta_value, dtype=torch.float))

penalty = Variable(torch.tensor(0.0, dtype=torch.float))


def f_beta(x):
    return torch.min(C2, x)


class Ifelse: # if target < test then body else orelse
    def __init__(self, target, test, body, orelse):
        self.target_obj = target
        self.test = test
        self.body = body
        self.orelse = orelse
        self.property = 'ifelse'
        self.p0 = 1.0
        self.p1 = 1.0
        self.target = target.value
    
    def getProperty(self):
        return self.property
    
    def execute(self, symbol_table):
        if isinstance(parameter_dict[self.target_obj.name].value, domain.Interval):
            true_interval = domain.Interval(torch.max(parameter_dict[self.target_obj.name].value.left, n_infinity), torch.min(parameter_dict[self.target_obj.name].value.right, self.test))
            false_interval = domain.Interval(torch.max(parameter_dict[self.target_obj.name].value.left, self.test), torch.min(parameter_dict[self.target_obj.name].value.right, p_infinity))
            self.p0 = torch.min(C3, true_interval.getLength().div(parameter_dict[self.target_obj.name].value.getLength().mul(f_beta(beta))))
            self.p1 = torch.min(C3, false_interval.getLength().div(parameter_dict[self.target_obj.name].value.getLength().mul(f_beta(beta))))

            if len(self.body) == 1 and len(self.orelse) == 1: # no other stmt in if-else
                if self.body[0].getProperty() == 'assign' and self.orelse[0].getProperty() == 'assign':
                    if self.body[0].target.name == self.orelse[0].target.name:
                        # final_target = self.body[0].target.value
                        self.body[0].target = Parameter(self.body[0].target.name, Variable(torch.tensor(self.body[0].target.value.data.item(), dtype=torch.float)))
                        self.orelse[0].target = Parameter(self.orelse[0].target.name, Variable(torch.tensor(self.orelse[0].target.value.data.item(), dtype=torch.float)))
                        # print('before assign, self.body[0].target.value', self.body[0].target.value)
                        for stmt in self.body:
                            # print('<68--------- assign')
                            stmt.execute()
                            # print('self.body[0].target.value', self.body[0].target.value, self.orelse[0].target.value)
                        for stmt in self.orelse:
                            stmt.execute()
                            # print('self.body[0].target.value', self.body[0].target.value, self.orelse[0].target.value)
                        
                        # avg join
                        # print('<68, >=68', self.body[0].target.value, self.orelse[0].target.value)
                        final_target = self.body[0].target.value.mul(self.p0).add(self.orelse[0].target.value.mul(self.p1))
                        # print('update parameter avg join', self.body[0].target.name, final_target)
                        parameter_dict[self.body[0].target.name].value = final_target
                        # assume target is not interval
            #TODO: avg join for multiple if-else

        else:
            if parameter_dict[self.target_obj.name].value.data.item() < self.test.data.item():
                for stmt in self.body:
                    stmt.execute()
            else:
                for stmt in self.orelse:
                    stmt.execute()


class While:
    def __init__(self, target, test, body): # , orelse):
        self.target_obj = target
        self.test = test
        self.body = body
        # self.orelse = orelse
        self.property = 'while'
        self.target = target.value

    def getProperty(self):
        return self.property
    
    def execute(self):
        if isinstance(parameter_dict[self.target_obj.name].value, domain.Interval):
            pass
        else:
            # print('in while')
            while(parameter_dict[self.target_obj.name].value.data.item() < self.test.data.item()):
                print('current isOn', parameter_dict['isOn'].value)
                print('current x interval', parameter_dict['X_interval'].value.left, parameter_dict['X_interval'].value.right)
                # print(parameter_dict[self.target_obj.name].value, self.test)
                # print('isOn', parameter_dict['isOn'].value)
                for stmt in self.body:
                    stmt.execute()


class Assign:
    def __init__(self, target, value):
        self.target = target
        self.value = value
        self.property = 'assign'

    def getProperty(self):
        return self.property
    
    def execute(self):
        if isinstance(parameter_dict[self.target.name].value, domain.Interval):
            self.target.value.left = self.value(parameter_dict[self.target.name].value.left)
            self.target.value.right = self.value(parameter_dict[self.target.name].value.right)
            parameter_dict[self.target.name].value.left = self.target.value.left
            parameter_dict[self.target.name].value.right = self.target.value.right
            # print('current x interval', parameter_dict[self.target.name].value.left, parameter_dict[self.target.name].value.right)
            # domain.Interval(self.value(self.target.left), self.value(self.target.right))

        else:
            # print(self.target.name)
            self.target.value = self.value(parameter_dict[self.target.name].value)
            # print('in assign', self.target.value)
            parameter_dict[self.target.name].value = self.target.value
            # print(self.target)
            # print('i', i)

    
class Return:
    def __init__(self, value):
        self.value = value
        self.property = 'return'
    
    def getProperty(self):
        return self.property


class Assert: # assert(taraget < constraint)
    def __init__(self, target, constraint):
        self.target = target
        self.constraint = constraint
        self.property = 'assert'
    
    def getProperty(self):
        return self.property

    def execute(self):
        global penalty
        #TODO: calculate the global penalty
        if isinstance(self.target.value, domain.Interval):
            non_constraint_interval = domain.Interval(self.constraint, p_infinity)
            assert_interval = domain.Interval(torch.max(non_constraint_interval.left, self.target.value.left), torch.min(non_constraint_interval.right, self.target.value.right))
            cur_penalty = torch.min(C3, assert_interval.getLength().div(self.target.value.getLength().mul(f_beta(beta))))
            penalty = penalty.add(cur_penalty)
        else:
            # if the assertion no related to x-interval
            # penalty += 1.0
            penalty = penalty.add(C3)


def f0(x):
    return C0


def f1(x):
    return C0


def f2(x):
    return C1


def f4(x): # x-K*(x-lin)
    # print('x-K*(x-lin)')
    return x.sub(K.mul(x.sub(lin))) 


def f10(x):
    # print('x-K*(x-lin)+h')
    return x.sub(K.mul(x.sub(lin))).add(h)


def f6(x):
    return C_on


def f8(x):
    return C_off


def f15(x):
    # print('in f15 x', x)
    return x.add(C3)


def f17(x): #abs(x-ltarget)
    return torch.max(x.sub(ltarget), C0.sub(x.sub(ltarget)))


# reconstruct the program
# all the comparison is '<'
l0 = Assign(parameter_dict['i'], f0)
l1 = Assign(parameter_dict['isOn'], f1)
l4 = Assign(parameter_dict['X_interval'], f4)
l6 = Assign(parameter_dict['isOn'], f6)
l8 = Assign(parameter_dict['isOn'], f8)
l5 = Ifelse(parameter_dict['X_interval'], Theta[0], [l6], [l8])
l10 = Assign(parameter_dict['X_interval'], f10)
l12 = Assign(parameter_dict['isOn'], f6)
l14 = Assign(parameter_dict['isOn'], f8)
l11 = Ifelse(parameter_dict['X_interval'], Theta[1], [l12], [l14])
l3 = Ifelse(parameter_dict['isOn'], C2, [l4, l5], [l10, l11])
l15 = Assign(parameter_dict['i'], f15)
l16 = Assert(parameter_dict['X_interval'], C_constraint)
l2 = While(parameter_dict['i'], C1, [l3, l15, l16])
l17 = Assign(parameter_dict['X_interval'], f17)


program = [l0, l1, l2]

# l15.execute()
# print('i value', i)


# for theta_0 in range(7500, 8500, 100):
#     theta_0 = theta_0 * 1.0 / 100
#     Theta = Variable(torch.tensor(theta_0, dtype=torch.float))

#     for stmt in program:
#         stmt.execute()

#     print(parameter_dict['Xs_interval'].value.left, parameter_dict['X_interval'].value.right)

#     break


def distance_f(X, target):
    X_length = X.getLength()
    if target.data.item() < X.right.data.item() and target.data.item() > X.left.data.item():
        res = C0
    else:
        res = torch.max(target.sub(X.right), X.left.sub(target)).div(X_length)
    
    return res


if __name__ == "__main__":
    
    lr = 0.1
    epoch = 1000

    # global penalty
    penalty = Variable(torch.tensor(0.0, dtype=torch.float))
    
    for i in range(epoch):
        
        for stmt in program:
            stmt.execute()
        
        f = distance_f(parameter_dict['X_interval'].value, ltarget).add(penalty)

        dTheta = torch.autograd.grad(f, Theta, retain_graph=True)
        derivation = dTheta[0]

        print(f.data, Theta.data)

        if torch.abs(derivation) < epsilon:
            print(f.data, Theta.data)
        
        Theta.data += lr * derivation.data

    print(f.data, Theta.data)
        

















