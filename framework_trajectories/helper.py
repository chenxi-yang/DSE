import torch
import random
from torch.autograd import Variable

def var(i, requires_grad=False):
    # print(i)
    return Variable(torch.tensor(i, dtype=torch.double), requires_grad=requires_grad)

PI = var((3373259426.0 + 273688.0 / (1 << 21)) / (1 << 30))
PI_TWICE = PI.mul(var(2.0))
PI_HALF = PI.div(var(2.0))

# print(PI)
# print(torch.sin(PI))
# print(torch.sin(PI_TWICE))
# print(torch.sin(PI_HALF))
