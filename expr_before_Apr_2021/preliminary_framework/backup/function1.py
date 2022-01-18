import domain

import torch
import random
from torch.autograd import Variable


class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value



i = Variable(torch.tensor(0.0, dtype=torch.float))
isOn = Variable(torch.tensor(0.0, dtype=torch.float))
K = Variable(torch.tensor(0.1, dtype=torch.float))
h = Variable(torch.tensor(5.0, dtype=torch.float))
lin = Variable(torch.tensor(60.0, dtype=torch.float))
tOff = Variable(torch.tensor(81.0, dtype=torch.float))
tOn = Variable(torch.tensor(68.0, dtype=torch.float))

ltarget = Variable(torch.tensor(75.0, dtype=torch.float))
X_l = Variable(torch.tensor(55.0, dtype=torch.float))
X_r = Variable(torch.tensor(65.0, dtype=torch.float))
X = Variable(torch.tensor(random.uniform(55.0, 65.0), dtype=torch.float))
X_interval = domain.Interval(X_l, X_r) # x is a symbolic representation


parameter_dict = dict()
parameter_dict['i'] = Parameter('i', i)
parameter_dict['isOn'] = Parameter('isOn', isOn)
# parameter_dict['K'] = Parameter('K', K)
# parameter_dict['h'] = Parameter('h', h)
# parameter_dict['lin'] = Parameter('lin', lin)
# parameter_dict['tOn'] = Parameter('tOn', tOn)
# parameter_dict['ltarget'] = Parameter('ltarget', ltarget)
parameter_dict['X_interval'] = Parameter('X_interval', X_interval)
