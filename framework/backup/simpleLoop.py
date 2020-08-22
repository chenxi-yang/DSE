"""
x = [7,10]
while(x < theta){
    x = x + 5
}
assert(x < 41)
return abs(x - 38)
"""

import domain
from helper import *
from initialization import *

import torch
import random
from torch.autograd import Variable
import copy

n_infinity_value = -1000
p_infinity_value = 1000
n_infinity = Variable(torch.tensor(n_infinity_value, dtype=torch.float))
p_infinity = Variable(torch.tensor(p_infinity_value, dtype=torch.float))

X_l = Variable(torch.tensor(7.0, dtype=torch.float))
X_r = Variable(torch.tensor(10.0, dtype=torch.float))
X_interval = domain.Interval(X_l, X_r)

domain_dict = dict()
domain_dict['interval'] = X_interval
domain_dict['p'] = Variable(torch.tensor(1.0, dtype=torch.float))

parameter_dict = dict()
parameter_dict['x'] = list()
parameter_dict['x'].append(domain_dict)

result_list = list()

C = Variable(torch.tensor(5.0, dtype=torch.float))
C0 = Variable(torch.tensor(0.5, dtype=torch.float))
C1 = Variable(torch.tensor(2.0, dtype=torch.float))
C2 = Variable(torch.tensor(1.0, dtype=torch.float))
C3 = Variable(torch.tensor(0.0, dtype=torch.float))

beta_value = 0.00001
beta = Variable(torch.tensor(beta_value, dtype=torch.float))

target = Variable(torch.tensor(38, dtype=torch.float))

constraint_value = Variable(torch.tensor(41.0, dtype=torch.float))
assert_interval = domain.Interval(constraint_value, p_infinity)

penalty_weight = Variable(torch.tensor(10.0, dtype=torch.float))


def f_beta(x):
    return torch.max(C0, beta)


def f(x):
    return x.add(C)
    

def assign(target, value):
    return value(target)


def notempty(parameter_list, constraint):
    for interval in parameter_list:
        if interval['interval'].right.data.item() > constraint.left.data.item():
            return True
    return False


def avg_join(interval_list):
    p_out = Variable(torch.tensor(0.0, dtype=torch.float))
    c_out = Variable(torch.tensor(0.0, dtype=torch.float))
    max_p = Variable(torch.tensor(0.0, dtype=torch.float))
    res_l = p_infinity
    res_r = n_infinity

    for interval_dict in interval_list:
        p_out = p_out.add(interval_dict['p'])
        max_p = torch.max(max_p, interval_dict['p'])
        interval_dict['c'] = interval_dict['interval'].getCenter()
        interval_dict['length'] = interval_dict['interval'].getLength()

        c_out = c_out.add(interval_dict['p'].mul(interval_dict['c']))
    
    c_out = c_out.div(p_out)

    for interval_dict in interval_list:
        interval_dict['p'] = interval_dict['p'].div(max_p)
        interval_dict['length'] = interval_dict['p'].mul(interval_dict['length'])

        interval_dict['c'] = interval_dict['p'].mul(interval_dict['c']).add((C2.sub(interval_dict['p']).mul(c_out)))
        interval_dict['interval'].left = interval_dict['c'].sub(interval_dict['length'].div(C1))
        interval_dict['interval'].right = interval_dict['c'].add(interval_dict['length'].div(C1))

        res_l = torch.min(res_l, interval_dict['interval'].left)
        res_r = torch.max(res_r, interval_dict['interval'].right)
    
    res_interval = domain.Interval(res_l, res_r)

    return res_interval


def distance_f(X, target):
    X_length = X.getLength()
    if target.data.item() < X.right.data.item() and target.data.item() > X.left.data.item():
        res = C3
    else:
        res = torch.max(target.sub(X.right), X.left.sub(target)).div(X_length)
    
    return res


def penalty_f(res_interval, assert_interval):
    intersection = domain.Interval(torch.max(res_interval.left, assert_interval.left), torch.min(res_interval.right, assert_interval.right))
    return torch.min(C2, intersection.getLength().div(res_interval.getLength()))


def cal_func(Theta, parameter_dict):

    constraint_interval = domain.Interval(n_infinity, Theta)
    neg_constraint_interval = domain.Interval(Theta, p_infinity)
    neg_list = list()

    # print(len(parameter_dict['x']), parameter_dict['x'][0]['interval'].left)
    while(notempty(parameter_dict['x'], constraint_interval)): # out loop
        # split_list = list() # split by the constraint
        # pos_list = list() # in loop

        for idx, interval in enumerate(parameter_dict['x']):
            if interval['interval'].left.data.item() > constraint_interval.right.data.item():
                neg_list.append(interval)
                del parameter_dict['x'][idx]
            
            if interval['interval'].left.data.item() < constraint_interval.right.data.item() and interval['interval'].right.data.item() > constraint_interval.right.data.item():
                # across constraint
                neg_interval = domain.Interval(constraint_interval.right, interval['interval'].right)
                pos_interval = domain.Interval(interval['interval'].left, constraint_interval.right)

                interval_dict = dict()
                interval_dict['interval'] = neg_interval
                interval_dict['p'] = interval['p'].mul(torch.min(C2, neg_interval.getLength().div(interval['interval'].getLength().mul(f_beta(beta)))))

                neg_list.append(interval_dict)

                interval_dict = dict()
                interval_dict['interval'] = pos_interval
                interval_dict['p'] = interval['p'].mul(torch.min(C2, pos_interval.getLength().div(interval['interval'].getLength().mul(f_beta(beta)))))

                del parameter_dict['x'][idx]
                parameter_dict['x'].append(interval_dict)

            if interval['interval'].right.data.item() < constraint_interval.right.data.item():
                pass

        for idx, interval in enumerate(parameter_dict['x']):
            # add constant
            interval['interval'].left = interval['interval'].left.add(C)
            interval['interval'].right = interval['interval'].right.add(C)
            # add theta
            print('theta in use', Theta.data)
            # interval['interval'].left = interval['interval'].left.add(Theta)
            # interval['interval'].right = interval['interval'].right.add(Theta)
    
    res_interval = avg_join(neg_list)
    print('res', res_interval.left, res_interval.right)

    distance = distance_f(res_interval, target)
    penalty = penalty_f(res_interval, assert_interval)

    f = distance.add(penalty.mul(penalty_weight))

    return f


def ini():

    X_l = Variable(torch.tensor(7.0, dtype=torch.float))
    X_r = Variable(torch.tensor(10.0, dtype=torch.float))
    X_interval = domain.Interval(X_l, X_r)

    domain_dict = dict()
    domain_dict['interval'] = X_interval
    domain_dict['p'] = Variable(torch.tensor(1.0, dtype=torch.float))

    parameter_dict = dict()
    parameter_dict['x'] = list()
    parameter_dict['x'].append(domain_dict)

    return parameter_dict


if __name__ == "__main__":
    
    lr = 0.1
    epoch = 10000

    l = 30
    r = 40

    parameter_dict = ini()

    Theta = Variable(torch.tensor(random.uniform(l, r), dtype=torch.float), requires_grad=True)
    print(Theta.data)

    for i in range(epoch):
        f = cal_func(Theta, parameter_dict)

        try:
            dTheta = torch.autograd.grad(f, Theta, retain_graph=True)
            derivation = dTheta[0]
        except RuntimeError:
            if torch.abs(f) < epsilon:
                print(f.data, Theta.data)
                break

            Theta = Variable(torch.tensor(random.uniform(l, r), dtype=torch.float), requires_grad=True)
            parameter_dict = ini()
            continue

        dTheta = torch.autograd.grad(f, Theta, retain_graph=True)
        derivation = dTheta[0]

        print(f.data, Theta.data)

        if torch.abs(derivation) < epsilon:
            print(f.data, Theta.data)
            break
        
        Theta.data -= lr * derivation.data

        parameter_dict = ini()

    print(f.data, Theta.data)








        


