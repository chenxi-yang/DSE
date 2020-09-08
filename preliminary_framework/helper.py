import torch
from torch.autograd import Variable

epsilon_value = 0.00001
epsilon = Variable(torch.tensor(epsilon_value, dtype=torch.float))