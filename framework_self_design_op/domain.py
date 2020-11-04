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

    def setValue(self, x):
        res = Interval()
        res.left = x
        res.right = x
        return res
    
    # arithmetic
    def add(self, y):
        # self + y
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = self.left.add(y)
            res.right = self.right.add(y)
        else:
            res.left = self.left.add(y.left)
            res.right = self.right.add(y.right)
        return res

    def sub_l(self, y):
        # self - y
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = self.left.sub(y)
            res.right = self.right.sub(y)
        else:
            res.left = self.left.sub(y.right)
            res.right = self.right.sub(y.left)
        return res

    def sub_r(self, y):
        # y - self
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = y.sub(var(1.0).mul(self.right))
            res.right = y.sub(var(1.0).mul(self.left))
        else:
            res.left = y.left.sub(self.right)
            res.right = y.right.sub(self.left)
        return res

    def mul(self, y):
        # self * y
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = torch.min(self.right.mul(y), self.left.mul(y))
            res.left = torch.max(self.right.mul(y), self.left.mul(y))
        else:
            res.left = torch.min(torch.min(y.left.mul(self.left), y.left.mul(self.right)), torch.min(y.right.mul(self.left), y.right.mul(self.right)))
            res.right = torch.max(torch.max(y.left.mul(self.left), y.left.mul(self.right)), torch.max(y.right.mul(self.left), y.left.mul(self.right)))
        return res

    def div(self, y):
        # y/self
        # 1. tmp = 1/self 2. res = tmp * y
        res = Interval()
        tmp_interval = Interval()
        tmp_interval.left = var(1.0).div(self.right)
        tmp_interval.right = var(1.0).div(self.left)
        res = tmp_interval.mul(y)
        return res

    def exp(self):
        res = Interval()
        res.left = torch.exp(self.left)
        res.right = torch.exp(self.right)
        return res

    def cos(self):
        res = Interval()

        cache = Interval()
        cache.left = self.left
        cache.right = self.right

        def handleNegative(interval):
            if interval.left.data.item() < 0.0:
                if interval.left.data.item() == N_INFINITY.data.item():
                    interval.left = var(0.0)
                    interval.right = P_INFINITY
                else:
                    n = torch.floor_divide(var(-1.0).mul(interval.left), PI_TWICE)
                    interval.left = interval.left.add(PI_TWICE.mul(n))
                    interval.right = interval.right.add(PI_TWICE.mul(n))
            return interval
        cache = handleNegative(cache)
        n = torch.floor_divice(cache.left, PI_TWICE)
        t = cache.sub_l(PI_TWICE.mul(n))

        if t.getVolumn().data.item >= PI_TWICE.data.item():
            return Interval(-1.0, 1.0)
        
        # when t.left > PI same as -cos(t-pi)
        if t.left.data.item() >= PI.data.item():
            cosv = t.sub_l(PI).cos()
            return cosv.mul(var(-1.0))
        
        tl = torch.cos(t.right)
        tr = torch.cos(t.left)
        if t.right.data.item() <= PI.data.item():
            res.left = tl
            res.right = tr
            return res
        elif t.right.data.item() <= PI_TWICE.data.item():
            res.left = var(-1.0)
            res.right = torch.max(tl, tr)
            return res
        else:
            res.left = var(-1.0)
            res.right = var(1.0)
            return res

    def sin(self):
        return self.cos(self.sub_l(PI_HALF))

    def max(self, y):
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = torch.max(self.left, y)
            res.right = torch.max(self.right, y)
        else:
            res.left = torch.max(self.left, y.left)
            res.right = torch.max(self.right, y.right)
        return res
    
    def min(self, y):
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = torch.min(self.left, y)
            res.right = torch.min(self.right, y)
        else:
            res.left = torch.min(self.left, y.left)
            res.right = torch.min(self.right, y.right)
        return res


class Zonotope:
    def __init__(self, left=0.0, right=0.0):
        self.center = var((left + right)/2.0)
        self.alpha_i = list()

    def getInterval(self):
        l = self.center
        r = self.center
        for i in alpha_i:
            l = l.sub(torch.abs(i))
            r = l.add(torch.abs(i))
        
        interval = Interval()
        interval.left = l
        interval.right = r

        return interval
    
    def getIntervalLength(self):
        interval = self.getInterval()
        return interval.getVolumn()
    
    def getVolumn(self):
        return self.getIntervalLength()
    
        # arithmetic
    def add(self, y):
        # self + y
        pass

    def sub_l(self, y):
        # self - y
        pass

    def sub_r(self, y):
        # y - self
        pass

    def mul(self, y):
        # self * y
        pass

    def div(self, y):
        # y / self
        pass

    def exp(self):
        # e^(self)
        pass

    def sin(self):
        pass

    def cos(self):
        pass

    def max(self):
        pass
    
    def min(self):
        pass


if __name__ == "__main__":
    a = Interval()
    b = Interval()
    print(a.add(b).left, a.add(b).right)
    # c = Interval().setValue(var(1.0))
    # print(c.left, c.right)