"""
Definition of different domains
1. interval
2. disjunction of intervalss
3. octagon
4. zonotope
5. polyhedra

"""
from constants import *
import constants

import torch
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time

import sys

# for check
def show_value(x):
    if not TEST:
        return 
    if isinstance(x, torch.Tensor):
        print('value', x.data.item())
    elif isinstance(x, Interval):
        print('interval', x.left.data.item(), x.righst.data.item())

def show_op(x):
    if not TEST:
        return 
    print(x + ':')


def handleNegative(interval):
    # print('interval', interval.left, interval.right)
    # if interval.left.data.item() < 0.0:
    #     # print('check')
    #     if interval.left.data.item() == N_INFINITY.data.item():
    #         interval.left = var(0.0)
    #         interval.right = P_INFINITY
    #     else:
    #         # n = torch.floor_divide(var(-1.0).mul(interval.left), PI_TWICE)
    #         n = torch.ceil(var(-1.0).mul(interval.left).div(PI_TWICE))
    #         interval.left = interval.left.add(PI_TWICE.mul(n))
    #         interval.right = interval.right.add(PI_TWICE.mul(n))
    
    left_neg = interval.left < 0.0
    left_neg_inf = interval.left[left_neg] <= float(N_INFINITY)
    left_neg_finite = interval.left[left_neg] > float(N_INFINITY)
    interval.left[left_neg][left_neg_inf] = 0
    interval.right[left_neg][left_neg_inf] = float(P_INFINITY)

    n = torch.ceil(-interval.left[left_neg][left_neg_finite] / float(PI_TWICE))
    interval.left[left_neg][left_neg_finite] = interval.left[left_neg][left_neg_finite] + float(PI_TWICE) * n
    interval.right[left_neg][left_neg_finite] = interval.right[left_neg][left_neg_finite] + float(PI_TWICE) * n

    return interval

# interval domain

class Interval:
    def __init__(self, left=var(0.0), right=var(0.0)):
        self.left = left
        self.right = right
    
    # for the same api
    def getInterval(self):
        res = Interval()
        res.left = self.left
        res.right = self.right
        return res
    
    def setInterval(self, l, r):
        res = Interval()
        res.left = l
        res.right = r
        return res
    
    def new(self, left, right):
        return self.__class__(left, right)
    
    def in_other(self, other):
        return torch.logical_and(self.left >= other.left, self.right <= other.right)
    
    def clone(self):
        return self.new(self.left.clone(), self.right.clone())

    def getBox(self):
        return Box(self.getCenter(), self.getDelta())
    
    def getLength(self):
        if self.right.data.item() < self.left.data.item():
            return var(0.0)
        else:
            # print(f"in getLength: {self.right}, {self.left}")
            # print(f"in getLength: {self.right.sub(self.left)}")
            return self.right.sub(self.left)
        
    def getVolumn(self):
        if self.right.data.item() < self.left.data.item():
            return var(0.0)
        else:
            return torch.max(EPSILON, (self.right.sub(self.left)))
    
    def split(self, partition):
        domain_list = list()
        unit = self.getVolumn().div(var(partition))
        for i in range(partition):
            new_domain = Interval()
            new_domain.left = self.left.add(var(i).mul(unit))
            new_domain.right = self.left.add(var(i + 1).mul(unit))
            domain_list.append(new_domain)
            # print('in split', new_domain.left, new_domain.right)
        return domain_list

    def getCenter(self):
        # C = var(2.0)
        return (self.left.add(self.right)).div(2.0)
    
    def getDelta(self):
        return (self.right.sub(self.left)).div(2.0)

    def equal(self, interval_2):
        if interval_2 is None:
            return False
        if interval_2.left.data.item() == self.left.data.item() and interval_2.right.data.item() == self.right.data.item():
            return True
        else:
            return False

    def isEmpty(self):
        # print(self.right, self.left)
        # print(f"judge empty: {self.left.data.item(), self.right.data.item()}")
        if self.right.data.item() < self.left.data.item():
            return True
        else:
            return False
    
    def isPoint(self):
        if float(self.right) == float(self.left): # or abs(self.right.data.item() - self.left.data.item()) < EPSILON.data.item():
            return True
        else:
            return False

    def setValue(self, x):
        res = Interval()
        res.left = x
        res.right = x
        return res
    
    def soundJoin(self, other):
        # if debug:
        #     r = torch.cuda.memory_reserved(0) 
        #     a = torch.cuda.memory_allocated(0)
        #     print(f"soundJoin, before, cuda memory reserved: {r}, allocated: {a}")
        res = self.new(torch.min(self.left, other.left), torch.max(self.right, other.right))
        # if debug:
        #     r = torch.cuda.memory_reserved(0) 
        #     a = torch.cuda.memory_allocated(0)
        #     print(f"soundJoin, after cuda memory reserved: {r}, allocated: {a}")
        return res
    
    def smoothJoin(self, other, alpha_prime_1, alpha_prime_2, alpha_1, alpha_2):
        c1, c2 = self.getCenter(), other.getCenter()
        delta1, delta2 = self.getDelta(), other.getDelta()
        c_out = (alpha_1 * c1 + alpha_2 * c2) / (alpha_1 + alpha_2)
        new_c1, new_c2 = alpha_prime_1 * c1 + (1 - alpha_prime_1) * c_out, alpha_prime_2 * c2 + (1 - alpha_prime_2) * c_out
        new_delta1, new_delta2 = alpha_prime_1 * delta1, alpha_prime_2 * delta2
        new_left = torch.min(new_c1 - new_delta1, new_c2 - new_delta2)
        new_right = torch.max(new_c1 + new_delta1, new_c1 + new_delta2)
        res = self.new(new_left, new_right)

        return res

    def getZonotope(self):
        res = Zonotope()
        res.center = (self.left.add(self.right)).div(var(2.0))
        res.alpha_i[0] = (self.right.sub(self.left)).div(var(2.0))
        return res
    
    # arithmetic
    def add(self, y):
        # self + y
        show_op('add')
        show_value(self)
        show_value(y)

        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = self.left.add(y)
            res.right = self.right.add(y)
        else:
            # print(res.left, y.left)
            res.left = self.left.add(y.left)
            res.right = self.right.add(y.right)

        show_value(res)
        return res

    def sub_l(self, y):
        # self - y
        show_op('sub_l')
        show_value(self)
        show_value(y)

        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = self.left.sub(y)
            res.right = self.right.sub(y)
        else:
            res.left = self.left.sub(y.right)
            res.right = self.right.sub(y.left)
        
        show_value(res)
        # if debug:
        #     print(f"#sub_l# size of res:  {sys.getsizeof(res), sys.getsizeof(res.left), sys.getsizeof(res.right)}")
        
        return res

    def sub_r(self, y):
        # y - self
        show_op('sub_r')
        show_value(y)
        show_value(self)
        
        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = y.sub(var(1.0).mul(self.right))
            res.right = y.sub(var(1.0).mul(self.left))
        else:
            res.left = y.left.sub(self.right)
            res.right = y.right.sub(self.left)
        
        show_value(res)
        # if debug:
        #     print(f"#sub_r# size of res:  {sys.getsizeof(res), sys.getsizeof(res.left), sys.getsizeof(res.right)}")
        return res

    def mul(self, y):
        # self * y
        # show_op('mul')
        # show_value(y)
        # show_value(self)
        # if debug:
        #     r1 = torch.cuda.memory_reserved(0) 
        #     a1 = torch.cuda.memory_allocated(0)
        #     print(f"#interval mul, ini#, memory: {a1}")
        #     print(f"in interval mul: self:{self.left, self.right}")
        #     print(f"in interval mul: y:{y}")

        res = Interval()
        # if debug:
        #     r2 = torch.cuda.memory_reserved(0) 
        #     a2 = torch.cuda.memory_allocated(0)
        #     print(f"#interval mul, ini Interval()#, memory: {a2}, {a2-a1}")
        if isinstance(y, torch.Tensor):
            # print(f"size of l, r of  res.left: {get_size(self.right.mul(y))},  {get_size(self.left.mul(y))}")
            res.left = torch.min(self.right.mul(y), self.left.mul(y))
            # res.left = torch.min(self.right.mul(y), self.left.mul(y))
            # tmp = self.right.mul(y)
            # if debug:
            #     a3 = torch.cuda.memory_allocated(0)
            #     print(f"#interval mul, res.left: {res.left}#, memory: {a3}, {a3-a1}")
            res.right = torch.max(self.right.mul(y), self.left.mul(y))
            # if debug:
            #     a4 = torch.cuda.memory_allocated(0)
            #     print(f"#interval mul, res.right: {res.right}#, memory: {a4}, {a4-a1}")
        else:
            res.left = torch.min(torch.min(y.left.mul(self.left), y.left.mul(self.right)), torch.min(y.right.mul(self.left), y.right.mul(self.right)))
            # if debug:
            #     a3 = torch.cuda.memory_allocated(0)
            #     print(f"#interval mul, res.left: {res.left}#, memory: {a3}, {a3-a1}")
            res.right = torch.max(torch.max(y.left.mul(self.left), y.left.mul(self.right)), torch.max(y.right.mul(self.left), y.right.mul(self.right)))
            # if debug:
            #     a4 = torch.cuda.memory_allocated(0)
                # print(f"#interval mul, res.right: {res.right}#, memory: {a4}, {a4-a1}")
        show_value(res)
        # if debug:
        #     print(f"#mul# size of res:  {get_size(res.left), get_size(res.right)}")
        #     exit(0)
        return res

    def div(self, y):
        # y/self
        # 1. tmp = 1/self 2. res = tmp * y
        show_op('div')
        show_value(y)
        show_value(self)

        res = Interval()
        tmp_interval = Interval()
        tmp_interval.left = var(1.0).div(self.right)
        tmp_interval.right = var(1.0).div(self.left)
        res = tmp_interval.mul(y)
        
        show_value(res)
        return res

    def exp(self):
        show_op('exp')
        show_value(self)

        # print(f'DEGUB:, exp', self.left, self.right)

        res = Interval()
        res.left = torch.exp(self.left)
        res.right = torch.exp(self.right)

        show_value(res)
        return res

    def cos(self):
        # show_cuda_memory(f"[cos] ini")
        cache = Interval(self.left, self.right)

        cache = handleNegative(cache)
        
        t = cache.fmod(PI_TWICE)
        del cache
        torch.cuda.empty_cache()
        # show_cuda_memory(f"[cos] before volume")
        if float(t.getVolumn()) >= float(PI_TWICE):
            # print('volume', t.getVolumn())
            res = Interval(var_list([-1.0]), var_list([1.0]))
            # show_cuda_memory(f"[cos] after 1 ")
            # show_value(res)
            # return res
        # when t.left > PI same as -cos(t-pi)
        elif float(t.left) >= float(PI):
            cosv = (t.sub_l(PI)).cos()
            res = cosv.mul(var_list([-1.0]))
            # show_cuda_memory(f"[cos] after left PI")
        else:
            tl = torch.cos(t.right)
            tr = torch.cos(t.left)
            if float(t.right) <= float(PI.data.item()):
                res = Interval(tl, tr)
            elif float(t.right) <= float(PI_TWICE):
                res = Interval(var_list([-1.0]), torch.max(tl, tr))
            else:
                res = Interval(var_list([-1.0]), var_list([1.0]))
            # show_cuda_memory(f"[cos] after else")
                
        # del cache
        del t
        torch.cuda.empty_cache()

        return res

    def sin(self):
        return self.sub_l(PI_HALF).cos()

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
    
    def sqrt(self):
        res = Interval()
        res.left = torch.sqrt(self.left)
        res.right = torch.sqrt(self.right)
        return res
    
    def fmod(self, y):
        # y is PI2

        #! not reasonable for batch TODO
        if isinstance(y, torch.Tensor):
            y_interval = Interval()
            y_interval = y_interval.setValue(y)
        else:
            y_interval = y
        
        if self.left.data.item() < 0.0:
            yb = y_interval.left
        else:
            yb = y_interval.right
        n = self.left.div(yb)
        if(n.data.item() <= 0.0): 
            n = torch.ceil(n)
        else:
            n = torch.floor(n)
        tmp_1 = y_interval.mul(n)

        res = self.sub_l(tmp_1)
        
        return res


class Box():
    def __init__(self, c, delta):
        self.c = c
        self.delta = delta
    
    def new(self, c, delta):
        # if debug:
        #     a1 = torch.cuda.memory_allocated(0)
        res = self.__class__(c, delta)
        # if debug:
        #     a2 = torch.cuda.memory_allocated(0)
        #     print(f"new box: memory cost: {a2 - a1}")
        return res
    
    def clone(self):
        return self.new(self.c.clone(), self.delta.clone())
    
    def check_in(self, other):
        # check: other in self (other.left >= self.left and other.right <= self.right)
        self_left = self.c - self.delta
        self_right = self.c + self.delta
        other_left = other.c - other.delta
        other_right = other.c + other.delta
        
        left_cmp = torch.ge(other_left, self_left)
        if False in left_cmp:
            return False
        right_cmp = torch.ge(self_right, other_right)
        if False in right_cmp:
            return False
        return True
    
    def select_from_index(self, dim, idx):
        return self.new(torch.index_select(self.c, dim, idx), torch.index_select(self.delta, dim, idx))

    def set_from_index(self, idx, other):
        # print(f"c: {self.c}, idx:{idx}")
        # print(f"c[idx]: {self.c[idx]}, other: {other.c, other.delta}")
        self.c[idx] = other.c
        self.delta[idx] = other.delta
        # print(f"c: {self.c}")
        return 
    
    def set_value(self, value):
        # print(f"value: {value}")
        return self.new(value, var(0.0))
    
    def sound_join(self, other):
        l1, r1 = self.c - self.delta, self.c + self.delta
        l2, r2 = other.c - other.delta, other.c + other.delta
        l = torch.min(l1, l2)
        r = torch.max(r1, r2)
        # if debug:
        #     r = torch.cuda.memory_reserved(0) 
        #     a = torch.cuda.memory_allocated(0)
        #     print(f"box sound_join, before, cuda memory reserved: {r}, allocated: {a}")
        res = self.new((r + l) / 2, (r - l) / 2)
        # if debug:
        #     r = torch.cuda.memory_reserved(0) 
        #     a = torch.cuda.memory_allocated(0)
        #     print(f"box sound_join, after, cuda memory reserved: {r}, allocated: {a}")
        return res
        
    def getRight(self):
        return self.c.add(self.delta)
    
    def getLeft(self):
        return self.c.sub(self.delta)
    
    def getInterval(self):
        res = Interval(self.c.sub(self.delta), self.c.add(self.delta))
        return res
    
    def matmul(self, other):
        # print(f"in matmul, self.c: {self.c.shape}, self.delta: {self.delta.shape}, other: {other.shape}")
        return self.new(self.c.matmul(other), self.delta.matmul(other.abs()))
    
    def add(self, other):
        if isinstance(other, torch.Tensor):
            c, d = self.c.add(other), self.delta
            res = self.new(c, d)
        else:
            c, d = self.c.add(other.c), self.delta + other.delta
            res = self.new(c, d)
        return res
            
    def sub_l(self, other): # self - other
        if isinstance(other, torch.Tensor):
            return self.new(self.c.sub(other), self.delta)
        else:
            return self.new(self.c.sub(other.c), self.delta + other.delta)
    
    def sub_r(self, other): # other - self
        if isinstance(other, torch.Tensor):
            return self.new(other.sub(self.c), self.delta)
        else:
            return self.new(other.c.sub(self.c), self.delta + other.delta)
    
    def mul(self, other):
        interval = self.getInterval()
        if isinstance(other, torch.Tensor):
            pass
        else:
            other = other.getInterval()
        res_interval = interval.mul(other)
        res = res_interval.getBox()
        return res
    
    def cos(self):
        #TODO: only for box, not for zonotope
        # todo: batch this
        # For batch:
        B = self.c.shape[0]
        if len(self.c.shape) > 1:
            c, delta = torch.tensor([]), torch.tensor([])
            # show_cuda_memory(f"before B")
            for i in range(B):
                # show_cuda_memory(f"-------ini B")
                new_c, new_delta = self.c[i], self.delta[i]
                new_box = self.new(new_c, new_delta)
                interval = new_box.getInterval()
                # show_cuda_memory(f"before cos")
                res_interval = interval.cos()
                # show_cuda_memory(f"after cos")
                get_box = res_interval.getBox()
                # print(c)
                if c.shape[0] == 0:
                    c, delta = get_box.c, get_box.delta
                else:
                    if len(get_box.c.shape) != len(c.shape) and len(c.shape) >= 1:
                        get_box.c, get_box.delta = get_box.c.unsqueeze(0), get_box.delta.unsqueeze(0)
                    c, delta = torch.cat((c, get_box.c), 0), torch.cat((delta, get_box.delta), 0)
            if len(c.shape) == 1:
                c, delta = c.unsqueeze(1), delta.unsqueeze(1)
            del self.c
            del self.delta
            return self.new(c, delta)
        else:
            # TODO: support batch and double check
            interval = self.getInterval()
            res_interval = interval.cos()
            res = res_interval.getBox()
            return res
    
    def exp(self):
        a = self.delta.exp()
        b = (-self.delta).exp()
        return self.new(self.c.exp().mul((a+b)/2), self.c.exp().mul((a-b)/2))
    
    def div(self, other): # other / self
        interval = self.getInterval()
        res_interval = interval.div(other)
        return res_interval.getBox()
    
    def sigmoid(self): # monotonic function
        tp = torch.sigmoid(self.c + self.delta)
        bt = torch.sigmoid(self.c - self.delta)
        # print(f"in sigmoid, tp: {tp}, bt: {bt}")
        return self.new((tp + bt)/2, (tp - bt)/2)
    
    def tanh(self): # monotonic function
        tp = torch.tanh(self.c + self.delta)
        bt = torch.tanh(self.c - self.delta)
        return self.new((tp + bt)/2, (tp - bt)/2)
    
    def relu(self): # monotonic function
        tp = F.relu(self.c + self.delta)
        bt = F.relu(self.c - self.delta)        
        return self.new((tp + bt)/2, (tp - bt)/2)
    
    def sigmoid_linear(self, sig_range):
        a = var(0.5/sig_range)
        b = var(0.5)
        x = self.mul(a).add(b)
        tp = torch.clamp(x.c + x.delta, 0, 1)
        bt = torch.clamp(x.c - x.delta, 0, 1)

        return self.new((tp + bt)/2, (tp - bt)/2)


class Zonotope:
    def __init__(self, left=0.0, right=0.0):
        self.center = var((left + right)/2.0)
        self.alpha_i = list([var((right - left)/2.0)])

    def getInterval(self):
        # print('c, alpha0', self.center, self.alpha_i)
        l = self.center
        r = self.center
        for i in self.alpha_i:
            l = l.sub(torch.abs(i))
            r = r.add(torch.abs(i))
        
        interval = Interval()
        interval.left = l
        interval.right = r
        # print('-------end c, alpha0', self.center, self.alpha_i[0])
        # print('----====l, r', l, r)
        return interval
    
    def getIntervalLength(self):
        interval = self.getInterval()
        return interval.getVolumn()

    def getLength(self):
        return self.getIntervalLength()

    def getVolumn(self):
        return self.getIntervalLength()
    
    def split(self, partition):
        domain_list = list()
        tmp_self = self.getInterval()
        unit = tmp_self.getVolumn().div(var(partition))
        for i in range(partition):
            new_domain = Interval()
            new_domain.left = tmp_self.left.add(var(i).mul(unit))
            new_domain.right = tmp_self.left.add(var(i + 1).mul(unit))
            domain_list.append(new_domain.getZonotope())
        return domain_list
    
    def getCoefLength(self):
        return len(self.alpha_i)
    
    def setValue(self, x):
        res = Zonotope()
        res.center = x
        res.alpha_i = list()
        return res
    
        # arithmetic
    def add(self, y):
        # self + y
        res = Zonotope()
        if isinstance(y, torch.Tensor):
            res.center = self.center.add(y)
            res.alpha_i = [i.add(y) for i in self.alpha_i]
        else:
            res.center = self.center.add(y.center)
            l1 = self.getCoefLength()
            l2 = y.getCoefLength()
            res_l = res.getCoefLength()
            largest_l = max(l1, l2)
            shortest_l = min(l1, l2)
            for i in range(largest_l - res_l):
                res.alpha_i.append(var(0.0)) # take spaces

            for i in range(shortest_l):
                res.alpha_i[i] = self.alpha_i[i].add(y.alpha_i[i])
            if l1 < l2:
                for i in range(l1, l2):
                    res.alpha_i[i] = y.alpha_i[i]
            else:
                for i in range(l2, l1):
                    res.alpha_i[i] = self.alpha_i[i]
        # print('after add', res.getInterval().left, res.getInterval().right)
        if isinstance(y, torch.Tensor):
            res = self.getInterval().add(y).getZonotope()
        else:
            res = self.getInterval().add(y.getInterval()).getZonotope()
        return res

    def sub_l(self, y):
        # self - y
        res = Zonotope()
        if isinstance(y, torch.Tensor):
            res.center = self.center.sub(y)
            res.alpha_i = [i.sub(y) for i in self.alpha_i]
        else:
            res.center = self.center.sub(y.center)
            l1 = self.getCoefLength()
            l2 = y.getCoefLength()
            res_l = res.getCoefLength()
            largest_l = max(l1, l2)
            shortest_l = min(l1, l2)
            for i in range(largest_l - res_l):
                res.alpha_i.append(var(0.0)) # take spaces

            for i in range(shortest_l):
                res.alpha_i[i] = self.alpha_i[i].sub(y.alpha_i[i])
            if l1 < l2:
                for i in range(l1, l2):
                    res.alpha_i[i] = var(0.0).sub(y.alpha_i[i])
            else:
                for i in range(l2, l1):
                    res.alpha_i[i] = self.alpha_i[i]
        # print('after sub_l', res.getInterval().left, res.getInterval().right)

        if isinstance(y, torch.Tensor):
            res = self.getInterval().sub_l(y).getZonotope()
        else:
            res = self.getInterval().sub_l(y.getInterval()).getZonotope()
        return res

    def sub_r(self, y):
        # y - self

        res = Zonotope()
        if isinstance(y, torch.Tensor):
            res.center = y.sub(self.center)
            res.alpha_i = [y.sub(i) for i in self.alpha_i]
        else:
            res.center = y.center.sub(self.center)
            l1 = self.getCoefLength()
            l2 = y.getCoefLength()
            res_l = res.getCoefLength()
            largest_l = max(l1, l2)
            shortest_l = min(l1, l2)
            for i in range(largest_l - res_l):
                res.alpha_i.append(var(0.0)) # take spaces

            for i in range(shortest_l):
                res.alpha_i[i] = y.alpha_i[i].sub(self.alpha_i[i])
            if l1 < l2:
                for i in range(l1, l2):
                    res.alpha_i[i] = y.alpha_i[i]
            else:
                for i in range(l2, l1):
                    res.alpha_i[i] = var(0.0).sub(self.alpha_i[i])
        # print('after sub_r', res.getInterval().left, res.getInterval().right)
        if isinstance(y, torch.Tensor):
            res = self.getInterval().sub_r(y).getZonotope()
        else:
            res = self.getInterval().sub_r(y.getInterval()).getZonotope()
        return res

    def mul(self, y):

        res = Zonotope()
        if isinstance(y, torch.Tensor):
            res.center = self.center.mul(y)
            res.alpha_i = [i.mul(y) for i in self.alpha_i]

            res = self.getInterval().mul(y).getZonotope()
        else:
            res = self.getInterval().mul(y.getInterval()).getZonotope()
        return res


    def div(self, y):
        tmp_res = self.getInterval()
        tmp_res = tmp_res.div(var(1.0))
        res = tmp_res.getZonotope().mul(y)
        
        return res

    def exp(self):
        tmp_res = self.getInterval()
        res = tmp_res.exp().getZonotope()

        return res

    def sin(self):
        tmp_res = self.getInterval()
        res = tmp_res.sin().getZonotope()
        return res

    def cos(self):
        tmp_res = self.getInterval()
        res = tmp_res.cos().getZonotope()
        return res

    def max(self, y):
        res = Zonotope()
        res.alpha_i = list()

        tmp_interval = self.getInterval()
        a = tmp_interval.left
        b = tmp_interval.right

        if isinstance(y, torch.Tensor):
            if b.data.item() >= y.data.item():
                res.center = self.center
                for i in self.alpha_i:
                    res.alpha_i.append(self.alpha_i[i])
            else:
                return Zonotope(y.data.item(), y.data.item())
        else:
            l1 = self.getCoefLength()
            l2 = y.getCoefLength()
            for i in range(max(l1, l2)):
                res.alpha_i.append(var(0.0))
            
            tmp_interval_2 = y.getInterval()
            m = tmp_interval_2.left
            n = tmp_interval_2.right

            if b.data.item() >= n.data.item():
                res.center = self.center
                for i in range(l1):
                    res.alpha_i[i] = self.alpha_i[i]
            else:
                res.center = y.center
                for i in range(l2):
                    res.alpha_i[i] = y.alpha_i[i]
        
        # 
        tmp_res =(self.getInterval().max(y.getInterval())).getZonotope()
        return tmp_res

    def min(self, y):
        res = Zonotope()
        res.alpha_i = list()

        tmp_interval = self.getInterval()
        a = tmp_interval.left
        b = tmp_interval.right

        if isinstance(y, torch.Tensor):
            if b.data.item() <= y.data.item():
                res.center = self.center
                for i in self.alpha_i:
                    res.alpha_i.append(self.alpha_i[i])
            else:
                return Zonotope(y.data.item(), y.data.item())
        else:
            l1 = self.getCoefLength()
            l2 = y.getCoefLength()
            for i in range(max(l1, l2)):
                res.alpha_i.append(var(0.0))
            
            tmp_interval_2 = y.getInterval()
            m = tmp_interval_2.left
            n = tmp_interval_2.right

            if b.data.item() <= n.data.item():
                res.center = self.center
                for i in range(l1):
                    res.alpha_i[i] = self.alpha_i[i]
            else:
                res.center = y.center
                for i in range(l2):
                    res.alpha_i[i] = y.alpha_i[i]
        
        # return res
        tmp_res = (self.getInterval().min(y.getInterval())).getZonotope()
        return tmp_res


class HybridZonotope:
    def __init__(self, head, beta, errors, **kargs): 
        self.head = head
        self.errors = errors
        self.beta = beta
        



if __name__ == "__main__":
    a = Interval()
    b = Interval()
    print(a.add(b).left, a.add(b).right)
    # c = Interval().setValue(var(1.0))
    # print(c.left, c.right)