"""
Definition of different domains
1. interval
2. disjunction of intervals
3. octagon
4. zonotope
5. polyhedra

"""
from helper import *

import torch
import random
from torch.autograd import Variable

# interval domain

class Interval:
    def __init__(self, left=0.0, right=0.0):
        self.left = Variable(torch.tensor(left, dtype=torch.float))
        self.right = Variable(torch.tensor(right, dtype=torch.float))
    
    def getLength(self):
        if self.right.data.item() < self.left.data.item():
            return Variable(torch.tensor(0.0, dtype=torch.float))
        else:
            return torch.max(epsilon, (self.right.sub(self.left)))
    
    def getVolumn(self):
        if self.right.data.item() < self.left.data.item():
            return Variable(torch.tensor(0.0, dtype=torch.float))
        else:
            return torch.max(epsilon, (self.right.sub(self.left)))
    
    def getLeft(self):
        return self.left
    
    def getRight(self):
        return self.right

    def getCenter(self):
        C = Variable(torch.tensor(2.0, dtype=torch.float))
        return (self.left.add(self.right)).div(C)

    def equal(self, interval_2):
        if interval_2 is None:
            return False
        if interval_2.left.data.item() == self.left.data.item() and interval_2.right.data.item() == self.right.data.item():
            return True
        else:
            return False

    def isEmpty(self):
        if self.right.data.item() < self.left.data.item():
            return True
        else:
            return False