import torch
from torch.autograd import Variable
import random
from collections import deque
from sympy import Symbol, Poly, solve
from sympy.solvers.inequalities import reduce_rational_inequalities
# from function1 import *
from function2 import *


# value = 11
# Theta = Variable(torch.tensor(value, dtype=torch.float), requires_grad=True)

X_l = Variable(torch.tensor(x_l, dtype=torch.float))
X_r = Variable(torch.tensor(x_r, dtype=torch.float))

C_l = Variable(torch.tensor(c_l, dtype=torch.float))
C_r = Variable(torch.tensor(c_r, dtype=torch.float))

n_infinity = Variable(torch.tensor(n_infinity_value, dtype=torch.float))
p_infinity = Variable(torch.tensor(p_infinity_value, dtype=torch.float))

epsilon = Variable(torch.tensor(0.0001, dtype=torch.float))
sat_constraint_distance = Variable(torch.tensor(1.0, dtype=torch.float))

c0 = Condition(0)
c1 = Condition(1)
c2 = Condition(2)
body_10 = Body(1, 0)
body_11 = Body(1, 1)
body_20 = Body(2, 0)
body_21 = Body(2, 1)

def complete_theta_l(Theta):
    f = n_infinity
    return f

def complete_theta_r(Theta):
    f = p_infinity
    return f

complete_set = (complete_theta_l, complete_theta_r)

condition_list = [c0, c1, c2]
body_list = [body_10, body_11, body_20, body_21]
expression_list = [expr_0, expr_1, expr_2]

condition_iter = iter(condition_list)
condition_tree = Tree()
condition_tree.root = Node(next(condition_iter))
# print('tree root', Tree.root)
# print('condition_tree, root', condition_tree.root)
fringe = deque([condition_tree.root])
while True:
    head = fringe.popleft()
    try:
        head.l = Node(next(condition_iter))
        # print('head.l.v', head.l.v)
        fringe.append(head.l)
        head.r = Node(next(condition_iter))
        fringe.append(head.r)
    except StopIteration:
        break


expression_iter = iter(expression_list)
expression_tree = Tree()
expression_tree.root = Node(next(expression_iter))
fringe = deque([expression_tree.root])
while True:
    head = fringe.popleft()
    try:
        head.l = Node(next(expression_iter))
        fringe.append(head.l)

        head.r = Node(next(expression_iter))
        fringe.append(head.r)
    except StopIteration:
            break


def split_interval(conditon_tree, complete_set):
    interval_list = list()
    root = condition_tree.root
    # print('root', root)

    def node_max(i, interval, node):
        def ret_f(Theta):
            f = torch.max(interval[i](Theta), node.v(Theta))
            return f 
        return ret_f
    
    def node_min(i, interval, node):
        def ret_f(Theta):
            f = torch.min(interval[i](Theta), node.v(Theta))
            return f
        return ret_f
        
    def visit_node(node, interval):
        # print('interval trace')
        # print(interval[0], interval[1])
        # print('node', node.v(Theta).data)
        if node.v is None:
            return
        if node.l is None:
            interval_list.append((interval[0], node_min(1, interval, node)))
            # interval_list.append((interval[0], torch.min(interval[1], node.v(Theta))))
        else:
            # print(interval[1], node.v)
            visit_node(node.l, (interval[0], node_min(1, interval, node)))
            # visit_node(node.l, (interval[0], torch.min(interval[1], node.v(Theta))))
        if node.r is None:
            interval_list.append((node_max(0, interval, node), interval[1]))
            # interval_list.append((torch.max(interval[0], node.v(Theta)), interval[1]))
        else:
            visit_node(node.r, (node_max(0, interval, node), interval[1]))
            # visit_node(node.r, (torch.max(interval[0], node.v(Theta)), interval[1]))

    visit_node(root, complete_set)

    return interval_list


def merge_branch_interval(interval_list, body_list, Theta):
    branch_list = list()

    for i, interval in enumerate(interval_list):
        branch_dict = dict()
        branch_dict['l'] = interval[0](Theta)
        branch_dict['r'] = interval[1](Theta)
        branch_dict['body'] = body_list[i]
        branch_list.append(branch_dict)

    return branch_list


def get_tensor_intersection(interval_l, interval_r, X_l, X_r):
    intersection_l = torch.max(interval_l, X_l)
    intersection_r = torch.min(interval_r, X_r)

    return intersection_l, intersection_r


def get_volumn(intersection_l, intersection_r, X_l, X_r):
    C = Variable(torch.tensor(0, dtype=torch.float))

    # max(min(b1, b2)-max(a1, a2), 0) [a1, b1] [a2, b2]
    volumn = torch.max(torch.min(intersection_r, X_r).sub(torch.max(intersection_l, X_l)), C).div(X_r.sub(X_l))

    return volumn


# do abstract interpretation for [intersection_l, intersection_r]
def get_output(intersection_l, intersection_r, body, Theta):
    #TODO: if theta is cross the gradient

    # compute the gradient of x, resulting in how to AI interval
    x_value = random.uniform(intersection_l.data, intersection_r.data)
    X = Variable(x_value.clone().detach(), requires_grad=True)

    body_f = body(X, Theta)
    dX = torch.autograd.grad(body_f, X)[0]

    if dX >= 0:
        input_l = intersection_l
        input_r = intersection_r
    else:
        input_l = intersection_r
        input_r = intersection_l
    
    output_l = body(input_l, Theta)
    output_r = body(input_r, Theta)

    return output_l, output_r


def get_distance(output_l, output_r, C_l, C_r, Theta):

    C = Variable(torch.tensor(0, dtype=torch.float))

    intersection_volumn = torch.max(torch.min(output_r, C_r).sub(torch.max(output_l, C_l)), C)
    # print('in get_distance, output_l, output_r', output_l, output_r)
    # print('intersection volumn', intersection_volumn.data)

    if intersection_volumn == 0.0: # no intersection, use hausdorff distance, 0-d_H(O_1, O_c)
        C1 = Variable(torch.tensor(0, dtype=torch.float))
        # print('output_l', output_l.data)
        # print('C_r', C_r.data)
        # print('4 values:')
        # print(torch.abs(output_l.sub(C_l)).data)
        # print(torch.abs(output_l.sub(C_r)).data)
        # print(torch.abs(output_r.sub(C_l)).data)
        # print(torch.abs(output_r.sub(C_r)).data)
        distance = C1.sub(torch.min(torch.abs(output_l.sub(C_l)), torch.min(torch.abs(output_l.sub(C_r)), torch.min(torch.abs(output_r.sub(C_l)), torch.abs(output_r.sub(C_r))))))
        # print('distance data in hausdorff', distance.data)
    else: # has intersection
        C1 = Variable(torch.tensor(1, dtype=torch.float))
        distance = torch.min(C1, intersection_volumn.div(output_r.sub(output_l)))
    
    # dtheta = torch.autograd.grad(distance, Theta, retain_graph=True)
    # print('distance derivation', dtheta[0])

    return distance


def cal_func(branch_list, X_l, X_r, C_l, C_r, Theta):
    C = Variable(torch.tensor(0, dtype=torch.float))
    f = C

    for branch_dict in branch_list:
        interval_l = branch_dict['l']
        interval_r = branch_dict['r']
        body = branch_dict['body']

        intersection_l, intersection_r = get_tensor_intersection(interval_l, interval_r, X_l, X_r)
        volumn = get_volumn(intersection_l, intersection_r, X_l, X_r)
        # print('intersection_l, intersection_r', intersection_l.data, intersection_r.data)
        # print('volumn data', volumn.data)

        if volumn.data == 0.0: # empty set, do not compute this branch
            continue

        output_l, output_r = get_output(intersection_l, intersection_r, body, Theta)
        # print('output_l, output_r', output_l.data, output_r.data)

        distance = get_distance(output_l, output_r , C_l, C_r, Theta)
        # print('distance.data', distance.data)

        # Fix Volumn
        fix_volumn = Variable(volumn.data.clone().detach())

        f = f.add(torch.mul(fix_volumn, distance))
        
        # dtheta = torch.autograd.grad(f, Theta, retain_graph=True)
        # print('f derivation', dtheta[0])
        # print('Theta.data', Theta.data)
    
    # print('last function')
    # dtheta = torch.autograd.grad(f, Theta, retain_graph=True)
    # print('f derivation', dtheta[0])
    # print('Theta.data', Theta.data)

    return f


def cal_theta(lr, epoch, condition_tree, complete_set, X_l, X_r, C_l, C_r, Theta):

    # initialization split complete set of theta

    interval_list = split_interval(condition_tree, complete_set)

    for i in range(epoch):

        # interval_list = split_interval(condition_tree, complete_set, Theta) #!

        # print('interval')
        # for interval in interval_list:
        #     print(interval[0], interval[1])

        branch_list = merge_branch_interval(interval_list, body_list, Theta)

        f = cal_func(branch_list, X_l, X_r, C_l, C_r, Theta)

        dTheta = torch.autograd.grad(f, Theta, retain_graph=True)
        derivation = dTheta[0]
        # print('derivattion', derivation)

        if torch.abs(derivation) < epsilon:
            return f.data, Theta.data
        Theta.data += lr * derivation.data
        # print('Theta.data', Theta.data)
    
    return f.data, Theta.data

# interpret res object
def interpret_result_interval(res):
    rel_op = res.rel_op
    op_l = res._args[0]
    op_r = res._args[1]

    if rel_op == '<=':
        if op_l == theta_value:
            res_l = n_infinity_value
            res_r = op_r
        else:
            res_l = op_l
            res_r = p_infinity_value
    else:
        if op_l == theta_value:
            res_l = op_r
            res_r = p_infinity_value
        else:
            res_l = n_infinity_value
            res_r = op_l

    return res_l, res_r


def find_theta_interval(x_l, x_r, expression, child): # check where theta: [X_l, X_r] \in interval
    #TODO: check whether \theta in expression
    if child == 'l':
        res = reduce_rational_inequalities([[x_r <= expression]], theta_value)
    else:
        res = reduce_rational_inequalities([[expression <= x_l]], theta_value)
        
    res_l, res_r = interpret_result_interval(res)

    return (res_l, res_r)


def add_space_theta_split(theta_split_list):
    theta_split_list.sort()
    new_theta_split_list = list()

    for i in range(len(theta_split_list) - 1):
        interval_1 = theta_split_list[i]
        interval_2 = theta_split_list[i + 1]
        new_theta_split_list.append((interval_1[1], interval_2[0]))
        # print('new_theta_split_list', new_theta_split_list)
    
    theta_split_list += new_theta_split_list

    return theta_split_list


def check_empty(interval):
    if interval[1] - interval[0] <= 0: #! rule out one point
        return True
    else:
        return False


def get_intersection(interval_1, interval_2):
    return (max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1]))


def split_theta(expression_tree, x_l, x_r):
    theta_split_list = list()
    root = expression_tree.root
    interval = (n_infinity_value, p_infinity_value)

    def visit_node(node, interval): # interval: the theta interval sat the current node and all the nodes above
        if node.v is None:
            return
        # if check_empty(interval):
        #     return
        expression = node.v
        interval_l = find_theta_interval(x_l, x_r, expression, 'l') # theta: x <= expression
        interval_r = find_theta_interval(x_l, x_r, expression, 'r') # theta: x >= expression
        theta_interval_l = get_intersection(interval_l, interval) # interval && theta
        theta_interval_r = get_intersection(interval_r, interval)

        # print('interval')
        # print(interval)
        # print('interval_l, interval_r, theta_interval_l, theta_interval_r')
        # print(interval_l, interval_r, theta_interval_l, theta_interval_r)
        
        if not check_empty(theta_interval_l): #TODO: check whether the \theta is in body
            if node.l is None:
                theta_split_list.append(theta_interval_l)
            else:
                visit_node(node.l, theta_interval_l)
        
        if not check_empty(theta_interval_r):
            if node.r is None:
                theta_split_list.append(theta_interval_r)
            else:
                visit_node(node.r, theta_interval_r)
    
    visit_node(root, interval)

    theta_split_list = add_space_theta_split(theta_split_list)

    return theta_split_list


if __name__ == "__main__":
    lr = 0.1
    epoch = 100000

    theta_split_list = split_theta(expression_tree, x_l, x_r)
    print('theta_split_list', theta_split_list)

    max_distance = Variable(torch.tensor(n_infinity, dtype=torch.float))
    res_theta = Variable(torch.tensor(random.uniform(n_infinity_value, p_infinity_value), dtype=torch.float), requires_grad=True)

    for theta_interval in theta_split_list:
        l = theta_interval[0]
        r = theta_interval[1]
        print('l, r', l, r)

        Theta = Variable(torch.tensor(random.uniform(l, r), dtype=torch.float), requires_grad=True)
        
        distance, theta = cal_theta(lr, epoch, condition_tree, complete_set, X_l, X_r, C_l, C_r, Theta)
        print('distance, theta', distance, theta)

        if distance > max_distance:
            max_distance = distance
            res_theta = theta
        
        if max_distance == sat_constraint_distance:
            break
    
    print('final result(distance, theta): ', distance, theta)


    # # Unit Test: interval
    # interval_list = split_interval(condition_tree, complete_set)
    # for interval in interval_list:
    #     print(interval[0], interval[1])








