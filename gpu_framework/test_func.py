
# from constants import *
from constants import *

# from gpu_DSE.train import *
# from gpu_DSE.evaluation import verification
# from gpu_DSE.data_generator import load_data

from args import *
from helper import *

import random
import time

class Interval:
    def __init__(self, left=var(0.0), right=var(0.0)):
        self.left = left
        self.right = right

    def mul(self, y):
        # self * y
        if debug:
            r1 = torch.cuda.memory_reserved(0) 
            a1 = torch.cuda.memory_allocated(0)
            print(f"#interval mul, ini#, memory: {a1}")
            print(f"in interval mul: self:{self.left, self.right}")
            print(f"in interval mul: y:{y}")

        res = Interval()
        if debug:
            r2 = torch.cuda.memory_reserved(0) 
            a2 = torch.cuda.memory_allocated(0)
            print(f"#interval mul, ini Interval()#, memory: {a2}, {a2-a1}")
        if isinstance(y, torch.Tensor):
            print(f"size of l, r of  res.left: {get_size(self.right.mul(y))},  {get_size(self.left.mul(y))}")
            res.left = torch.min(self.right.mul(y), self.left.mul(y))
            # res.left = torch.min(self.right.mul(y), self.left.mul(y))
            # tmp = self.right.mul(y)
            if debug:
                a3 = torch.cuda.memory_allocated(0)
                print(f"#interval mul, res.left: {res.left}#, memory: {a3}, {a3-a1}")
            res.right = torch.max(self.right.mul(y), self.left.mul(y))
            if debug:
                a4 = torch.cuda.memory_allocated(0)
                print(f"#interval mul, res.right: {res.right}#, memory: {a4}, {a4-a1}")
        else:
            res.left = torch.min(torch.min(y.left.mul(self.left), y.left.mul(self.right)), torch.min(y.right.mul(self.left), y.right.mul(self.right)))
            if debug:
                a3 = torch.cuda.memory_allocated(0)
                print(f"#interval mul, res.left: {res.left}#, memory: {a3}, {a3-a1}")
            res.right = torch.max(torch.max(y.left.mul(self.left), y.left.mul(self.right)), torch.max(y.right.mul(self.left), y.right.mul(self.right)))
            if debug:
                a4 = torch.cuda.memory_allocated(0)
                print(f"#interval mul, res.right: {res.right}#, memory: {a4}, {a4-a1}")
        if debug:
            print(f"#mul# size of res:  {get_size(res.left), get_size(res.right)}")
        return res

def tmp_func(a, b):
    print('before f', torch.cuda.memory_allocated(0))
    c = a.mul(b)
    print('after f', torch.cuda.memory_allocated(0))
    return c 

if __name__ == "__main__":
    # a = Interval(var(0.0, True), var(1.0, True))
    # b = Interval(var(1.0, True), var(2.0, True))
    a = Interval(var_list([0.0]), var([1.0]))
    b = Interval(var_list([1.0]), var([2.0]))
    c = tmp_func(a, b)
    print(torch.cuda.memory_allocated(0))







