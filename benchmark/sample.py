import torch
from torch.autograd import Variable
import random

interval_a, interval_b = -3, 10
constraint_a, constraint_b = -20, 60

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
        # print(f)
        # 5<=y<=60
        # dT = torch.autograd.grad(torch.max(torch.max(L.sub(torch.mul(C1, X1).add(T).add(C2)), (torch.mul(C1, X2).add(T).add(C2)).sub(R)), C3), T)
        # -20<=y<=60, y<=x^2
        X = Variable(torch.tensor(random.uniform(interval_a, interval_b)))
        dT = torch.autograd.grad(torch.max(torch.max(torch.max(L.sub(torch.mul(C1, X1).add(T).add(C2)), (torch.mul(C1, X2).add(T).add(C2)).sub(R)), torch.mul(C1, X).add(T).add(C2).sub(torch.mul(C1, X))), C3), T)
        
        derivation = dT[0]
        if derivation == 0:
            return T.data
        T.data -= lr * derivation.data
    return T.data

def sol():
    lr = 0.1 # 1e-1
    epoch = 1000
    # target_func = target_f(interval_a, interval_b)

    theta = cal_theta(lr, epoch)
    print(theta)

sol()



