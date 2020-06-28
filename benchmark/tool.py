import sys
sys.path.append('/Users/cxyang/anaconda3/lib/python3.7/site-packages/pydrogen')

import pydrogen
from typing import Any, List, Callable, TypeVar, NewType, Tuple
#interval

class Interval(object):
    """
    A half-open interval on the real number line.
    """
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return 'Interval(%f, %f)' % (self.lo, self.hi)

    def width(self):
        return self.hi - self.lo


class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """

        out = []  
        for i in sorted(intervals, key=lambda i: i.start):
            if out and i.start <= out[-1].end:
                out[-1].end = max(out[-1].end, i.end)
            else:
                out.append(i)
        return out

class IntervalAI(pydrogen.Typical):
    def Statements(self, ss): return ss.post()[-1] # Last statement.
    def Module(self, ss): return ss.post()
    def FunctionDef(self, ss): return ss.post()
    def Return(self, e): return e.post()

    # def True_(self): return 'Bool'
    # def False_(self): return 'Bool'
    # def BoolOp(self, es): return 'Bool'if frozenset(es.post()) == {'Bool'} else 'Error'
    # def Not(self, e): return 'Bool' if e.post() == 'Bool' else 'Error'

    def Assign(self, targets, e, context=None): 
        # print(dir(targets), dir(e))
        print('assign', e.post(context))
        return e.post(context)
    def Tuple(self, es): return es
    def Num(self, n): 
        print('n', n.post())
        return n.post() # return 'Int'
    def Add(self, e1, e2): 
        print('here')
        print(e1.post(), e2.post())
        if type(e1.post()) == Tuple and type(e2.post()) == Tuple:
            # print('here1')
            return Tuple(e1.post()[0] + e2.post()[0], e1.post()[1] + e2.post()[1])
        else:
            if type(e1.post()) == int and type(e2.post()) == int:
                # print('here int')
                # print(e1.post(), e2.post())
                return e1.post() + e2.post()
            return 'Error' 
        # return 'Int' if e1.post() == 'Int' and e2.post() == 'Int' else 'Error'
    # def Sub(self, e1, e2): return 'Int' if e1.post() == 'Int' and e2.post() == 'Int' else 'Error'
    # def Mult(self, e1, e2): return 'Int' if e1.post() == 'Int' and e2.post() == 'Int' else 'Error'
    # def USub(self, e): return 'Int' if e.post() == 'Int' else 'Error'

    # def Tuple(self, es):
    #     print('in tuple')
    #     x = Interval(a, b)
    #     return x
    def Name(self, id, context=None): 
        print('id, ctx', id, context)
        # print('context[id]', context[id])
        print('id.post', id.post())
        return id.post()
    def BinOp(self, e1, e2, context=None): 
        print('binop', e1.post(), e2.post()) 
        return e1.post(context) + e2.post(context)
    # def Call(self, func, args):


@IntervalAI
def test(x:int, theta:int):
    x = 1
    theta = 3
    x1 = x
    x2 = theta
    x3 = x1 + x2
    return x3

print(str(test.IntervalAI))

# @IntervalAI
# def ai_test(x:Tuple[float, float], theta:Tuple[float, float]):
#     return x + theta
# 
# print(str(ai_test((1, 2), (3, 3)).IntervalAI))

# @Ty
# def correct():
#     return -1 + 2 - 3 * 4

# print("The type of 'correct' is " + str(correct.Ty) + ".")

# @Ty
# def incorrect():
#     return 123 and False

# print("The type of 'incorrect' is " + str(incorrect.Ty) + ".")

# @Ty
# def branch(x):
#     if x > 0:
#         return int(x)
#     else:
#         return str(x)

# print("The type of 'branch' is " + str(branch.Ty) + ".")
