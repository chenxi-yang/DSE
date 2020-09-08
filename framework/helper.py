import torch
import random
from torch.autograd import Variable

def var(i, requires_grad=False):
    return Variable(torch.tensor(i, dtype=torch.float), requires_grad=requires_grad)

