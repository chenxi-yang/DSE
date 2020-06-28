import torch
from torch.autograd import Variable
import random

interval_a, interval_b = -3, 10
constraint_a, constraint_b = -20, 60
a2 = -1
b2 = 1

def target_f(a, b):
    B, F = 130, 10

    X1 = Variable(torch.tensor(interval_a))
    X2 = Variable(torch.tensor(interval_b))
    C1 = Variable(torch.tensor(3))
    C2 = Variable(torch.tensor(2))

    T = Variable(torch.randn(B, F))

    f = torch.max(C1 * X1 + T + C2, C1 * X2 + T + C2)

    return f


def cal_theta(lr, epoch):

    X1 = Variable(torch.tensor(interval_a))
    X2 = Variable(torch.tensor(interval_b))

    C1 = Variable(torch.tensor(3))
    C2 = Variable(torch.tensor(2))
    C3 = Variable(torch.tensor(0.0))

    L = Variable(torch.tensor(constraint_a))
    R = Variable(torch.tensor(constraint_b))

    T = Variable(torch.tensor(random.uniform(-20, 20)), requires_grad=True)
    # print(T)

    # f = torch.max(torch.mul(C1, X1).add(torch.mul(C2, T)).add(C2), torch.mul(C1, X2).add(T).add(C2))
    # f = T.pow(2)
    for i in range(epoch):
        
        X = Variable(torch.tensor(random.uniform(interval_a, interval_b)))
        dT = torch.autograd.grad(torch.max(torch.max(torch.max(L.sub(torch.mul(C1, X1).add(T).add(C2)), (torch.mul(C1, X2).add(T).add(C2)).sub(R)), torch.mul(C1, X).add(T).add(C2).sub(torch.mul(C1, X))), C3), T)
        
        derivation = dT[0]
        if derivation == 0:
            return T.data
        T.data -= lr * derivation.data
    return T.data


def select_interval(T):
    
    C1 = Variable(torch.tensor(-5, dtype=torch.float))
    C2 = Variable(torch.tensor(9, dtype=torch.float))
    C3 = Variable(torch.tensor(-1, dtype=torch.float))
    C4 = Variable(torch.tensor(1, dtype=torch.float))

    theta = T.data

    if theta >= 1/11:
        l = torch.mul(C1, T).add(T)
        r = torch.mul(C2, T).add(T)
    elif theta >= 0:
        l = torch.mul(C1, T).add(T)
        r = torch.mul(C3, T).add(C4)
    elif theta > -1/3:
        l = torch.mul(C2, T).add(T)
        r = torch.mul(C3, T).add(C4)
    elif theta > -6:
        print('here4, T.data', T.data)
        l = torch.mul(C2, T).add(T)
        r = torch.mul(C1, T).add(T)
        print('l-r', l.data, r.data)
    else:
        l = torch.mul(C3, T).add(C4)
        r = torch.mul(C3, T).add(C4)
    
    return l, r


def select_function(l, r, T):
    C1 = Variable(torch.tensor(a2, dtype=torch.float))
    C2 = Variable(torch.tensor(b2, dtype=torch.float))
    C3 = Variable(torch.tensor(1, dtype=torch.float))
    C4 = Variable(torch.tensor(0, dtype=torch.float))
    epsilon = Variable(torch.tensor(0.0001, dtype=torch.float))
    print(l.data, r.data)

    if min(max(l.data - b2, a2 - r.data), 0) == 0:
        print('empty')
        f = C4.sub(torch.min(torch.abs(C2.sub(r)), torch.min(torch.abs(C1.sub(l)), torch.min(torch.abs(C1.sub(r)), torch.abs(l.sub(C2))))))
    else:
        if(r.data - b2 > 0):
            if(l.data - a2) >= 0:
                f = torch.min(C3, torch.div(torch.max(C2.sub(l), epsilon), torch.max(r.sub(l), epsilon)))
            else:
                # print('here, c < ai output')
                # print('C2.sub(C1)', C2.sub(C1).data)
                # print('r.sub(l))', r.sub(l).data)
                f = torch.min(C3, torch.div(torch.max(C2.sub(C1), epsilon), torch.max(r.sub(l), epsilon)))
        else:
            if(l.data - a2 >= 0):
                print('c in lr!!!')
                f = torch.mul(C4, T)
            else:
                f = torch.min(C3, torch.div(r.sub(C1), torch.max(r.sub(l), epsilon)))
    
    return f

    
def ai_cal_theta(lr, epoch):

    # T = Variable(torch.tensor(random.uniform(-10, 10)), requires_grad=True)
    T = Variable(torch.tensor(random.uniform(-10, 10)), requires_grad=True)
    print('T.data', T.data)

    for i in range(epoch):
        l, r = select_interval(T)
        f = select_function(l, r, T)
        print('f.data', f.data)

        dT = torch.autograd.grad(f, T)
        derivation = dT[0]
        print('derivation', derivation)
        if derivation == 0:
            return T.data
        T.data += lr * derivation.data
        print('T.data', T.data)
    
    return T.data

def sol():
    lr = 1.0 # 1e-1
    epoch = 10000
    # target_func = target_f(interval_a, interval_b)

    # theta = cal_theta(lr, epoch)
    theta = ai_cal_theta(lr, epoch)
    print(theta)

sol()



