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

# for check
def show_value(x):
    if not TEST:
        return 
    if isinstance(x, torch.Tensor):
        print('value', x.data.item())
    elif isinstance(x, Interval):
        print('interval', x.left.data.item(), x.right.data.item())

def show_op(x):
    if not TEST:
        return 
    print(x + ':')

# interval domain

class Interval:
    def __init__(self, left=0.0, right=0.0):
        self.left = var(left)
        self.right = var(right)
    
    # for the same api
    def getInterval(self):
        res = Interval()
        res.left = self.left
        res.right = self.right
        return res
    
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
        return res

    def mul(self, y):
        # self * y
        show_op('mul')
        show_value(y)
        show_value(self)

        res = Interval()
        if isinstance(y, torch.Tensor):
            res.left = torch.min(self.right.mul(y), self.left.mul(y))
            res.right = torch.max(self.right.mul(y), self.left.mul(y))
        else:
            res.left = torch.min(torch.min(y.left.mul(self.left), y.left.mul(self.right)), torch.min(y.right.mul(self.left), y.right.mul(self.right)))
            res.right = torch.max(torch.max(y.left.mul(self.left), y.left.mul(self.right)), torch.max(y.right.mul(self.left), y.right.mul(self.right)))
        show_value(res)
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
        show_op('cos')
        show_value(self)
        
        res = Interval()

        cache = Interval()
        cache.left = self.left
        cache.right = self.right

        def handleNegative(interval):
            # print('interval', interval.left, interval.right)
            if interval.left.data.item() < 0.0:
                # print('check')
                if interval.left.data.item() == N_INFINITY.data.item():
                    interval.left = var(0.0)
                    interval.right = P_INFINITY
                else:
                    # n = torch.floor_divide(var(-1.0).mul(interval.left), PI_TWICE)
                    n = torch.ceil(var(-1.0).mul(interval.left).div(PI_TWICE))
                    interval.left = interval.left.add(PI_TWICE.mul(n))
                    interval.right = interval.right.add(PI_TWICE.mul(n))
            return interval

        cache = handleNegative(cache)
        # print('y_neg', cache.left, cache.right)
        # n = torch.floor_divide(cache.left, PI_TWICE)
        # t = cache.sub_l(PI_TWICE.mul(n))
        t = cache.fmod(PI_TWICE)
        # print('t', t.left, t.right)

        # print(type(t.getVolumn()), type(PI_TWICE))
        if t.getVolumn().data.item() >= PI_TWICE.data.item():
            # print('volume', t.getVolumn())
            res = Interval(-1.0, 1.0)
            show_value(res)
            return res
        
        # when t.left > PI same as -cos(t-pi)
        if t.left.data.item() >= PI.data.item():
            cosv = (t.sub_l(PI)).cos()
            res = cosv.mul(var(-1.0))
            show_value(res)
            return res
        
        tl = torch.cos(t.right)
        tr = torch.cos(t.left)
        if t.right.data.item() <= PI.data.item():
            res.left = tl
            res.right = tr
            show_value(res)
            return res
        elif t.right.data.item() <= PI_TWICE.data.item():
            res.left = var(-1.0)
            res.right = torch.max(tl, tr)
            show_value(res)
            return res
        else:
            res.left = var(-1.0)
            res.right = var(1.0)
            show_value(res)
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
        if isinstance(y, torch.Tensor):
            y_interval = Interval()
            y_interval = y_interval.setValue(y)
        else:
            y_interval = y
        
        if self.left.data.item() < 0.0:
            yb = y_interval.left
        else:
            yb = y_interval.right
        # print('self left', self.left)
        # print('yb', yb)
        n = self.left.div(yb)
        if(n.data.item() <= 0.0): 
            n = torch.ceil(n)
        else:
            n = torch.floor(n)
        # print('n', n)

        n_interval = Interval()
        n_interval = n_interval.setValue(n)
        # print(self.left, self.right)
        # print(y_interval.mul(n_interval).left, y_interval.mul(n_interval).right)
        return self.sub_l(y_interval.mul(n_interval))


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
        # print('in add', self.getInterval().left, self.getInterval().right)
        # print('self, c', self.center)
        # for i in self.alpha_i:
        #     print(i)

        # if isinstance(y, torch.Tensor):
        #     print('y', y)
        # else:
        #     print('y', y.getInterval().left, y.getInterval().right)
        # self + y
        res = Zonotope()
        if isinstance(y, torch.Tensor):
            res.center = self.center.add(y)
            res.alpha_i = [i.add(y) for i in self.alpha_i]
        else:
            res.center = self.center.add(y.center)
            l1 = self.getCoefLength()
            l2 = y.getCoefLength()
            # print('l1, l2', l1, l2)
            # print('self c', self.center)
            # for i in self.alpha_i:
            #     print(i)
            # print('y c', self.center)
            # for i in y.alpha_i:
            #     print(i)
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
        # print('in sub_l', self.getInterval().left, self.getInterval().right)
        # if isinstance(y, torch.Tensor):
        #     print('y', y)
        # else:
        #     print('y', y.getInterval().left, y.getInterval().right)
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
        # print('in sub_r', self.getInterval().left, self.getInterval().right)
        # if isinstance(y, torch.Tensor):
        #     print('y', y)
        # else:
        #     print('y', y.getInterval().left, y.getInterval().right)

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
        # self * y
        # print('in mul', self.getInterval().left, self.getInterval().right)
        # print('self, c', self.center)
        # for i in self.alpha_i:
        #     print(i)
        # if isinstance(y, torch.Tensor):
        #     print('y', y)
        # else:
        #     print('y', y.getInterval().left, y.getInterval().right)

        res = Zonotope()
        if isinstance(y, torch.Tensor):
            res.center = self.center.mul(y)
            res.alpha_i = [i.mul(y) for i in self.alpha_i]

            res = self.getInterval().mul(y).getZonotope()
        else:
            # res.center = self.center.mul(y.center)
            # res.alpha_i = list()
            
            # l1 = self.getCoefLength()
            # l2 = y.getCoefLength()
            # max_l = max(l1, l2)
            # # equalize the length
            # for i in range(l1, max_l):
            #     self.alpha_i.append(var(0.0))
            # for i in range(l2, max_l):
            #     y.alpha_i.append(var(0.0))
            
            # for i in range(max_l):
            #     tmp_coef = self.center.mul(y.alpha_i[i]).add(y.center.mul(self.alpha_i[i]))
            #     res.alpha_i.append(tmp_coef)
            
            # # last noise coef: uv, u = sum abs(x_i), v = sum abs(y_i)
            # u = var(0.0)
            # v = var(0.0)
            # for i in range(l1):
            #     u = u.add(torch.abs(self.alpha_i[i]))
            # for i in range(l2):
            #     v = v.add(torch.abs(y.alpha_i[i]))
            res = self.getInterval().mul(y.getInterval()).getZonotope()
            # res.alpha_i.append(u.mul(v))
        # print('after mul', res.getInterval().left, res.getInterval().right)
        return res


    def div(self, y):

        # print('in div', self.getInterval().left, self.getInterval().right)
        # if isinstance(y, torch.Tensor):
        #     print('y', y)
        # else:
        #     print('y', y.getInterval().left, y.getInterval().right)
        # # y / self
        # # use mini-range approximation
        # res = Zonotope()
        # l1 = self.getCoefLength()
        # res_l = res.getCoefLength()
        # if res_l < l1:
        #     for i in range(res_l, l1):
        #         res.alpha_i.append(var(0.0))

        # tmp_interval = self.getInterval()
        # m = tmp_interval.left
        # n = tmp_interval.right

        # if m.data.item() <= 0 and n.data.item() >= 0:
        #     return Zonotope(N_INFINITY.data.item(), P_INFINITY.data.item())

        # t1 = torch.abs(m)
        # t2 = torch.abs(n)

        # a = torch.min(t1, t2)
        # b = torch.max(t1, t2)

        # alpha = var(-1.0).div(b.mul(b))
        # # i = ((1/a)-alpha*a, 2/b), alpha_0 = i.mid() 
        # dzeta = (((var(1.0)/a).sub(alpha.mul(a))).add(var(2.0).div(b))).div(2.0)
        # delta = var(2.0).div(b).sub(dzeta)

        # if tmp_interval.left.data.item() < 0.0:
        #     dzeta = dzeta.mul(var(-1.0))
        
        # res.center = alpha.mul(self.center).add(dzeta)
        # for i in range(l1):
        #     res.alpha_i[i] = alpha.mul(self.alpha_i[i])
        # res.alpha_i.append(delta)
        # # res = 1/self
        
        # res = res.mul(y)

        # print('after div', res.getInterval().left, res.getInterval().right)
        tmp_res = self.getInterval()
        tmp_res = tmp_res.div(var(1.0))
        res = tmp_res.getZonotope().mul(y)
        
        return res

    def exp(self):
        # e^(self)
        # print('in exp', self.getInterval().left, self.getInterval().right)
        # res = Zonotope()
        # l1 = self.getCoefLength()
        # res_l = res.getCoefLength()
        # if res_l < l1:
        #     for i in range(res_l, l1):
        #         res.alpha_i.append(var(0.0))
        
        # tmp_interval = self.getInterval()
        # a = tmp_interval.left
        # b = tmp_interval.right

        # ea = torch.exp(a)
        # eb = torch.exp(b)

        # alpha = (eb.sub(ea)).div(b.sub(a))
        # xs = torch.log(alpha)
        # maxdelta = alpha.mul(xs.sub(var(1.0).sub(a))).add(ea)
        # dzeta = alpha.mul(var(1.0).sub(xs))
        # # get the error
        # delta = maxdelta.div(var(2.0))

        # res.center = alpha.mul(self.center).add(dzeta)
        # for i in range(l1):
        #     res.alpha_i[i] = alpha.mul(self.alpha_i[i])
        # res.alpha_i.append(delta)

        # print('after exp', res.getInterval().left, res.getInterval().right)

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


if __name__ == "__main__":
    a = Interval()
    b = Interval()
    print(a.add(b).left, a.add(b).right)
    # c = Interval().setValue(var(1.0))
    # print(c.left, c.right)