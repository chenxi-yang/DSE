import torch
import random
from torch.autograd import Variable

def var(i, requires_grad=False):
    # print(i)
    return Variable(torch.tensor(i, dtype=torch.double), requires_grad=requires_grad)

