"""
Definition of different domains
1. interval
2. disjunction of intervals
3. octagon
4. zonotope
5. polyhedra

"""
from helper import *
from constants import *

import torch
import random
from torch.autograd import Variable

# interval domain

class Interval:
    def __init__(self, left=0.0, right=0.0):
        self.left = var(left)
        self.right = var(right)
    
    def getLength(self):
        if self.right.data.item() < self.left.data.item():
            return var(0.0)
        else:
            return torch.max(EPSILON, (self.right.sub(self.left)))
    
    def getVolumn(self):
        if self.right.data.item() < self.left.data.item():
            return var(0.0)
        else:
            return torch.max(EPSILON, (self.right.sub(self.left)))
    
    def getLeft(self):
        return self.left
    
    def getRight(self):
        return self.right

    def getCenter(self):
        C = var(2.0)
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
    
    def isPoint(self):
        if self.right.data.item() == self.left.data.item():
            return True
        else:
            return False