import sys
sys.path.append('/Users/cxyang/anaconda3/lib/python3.7/site-packages/pydrogen')

import ast
import inspect
import pydrogen
import sympy
from sympy import O, symbols, oo

# some complexity classes using sympy's O notation
linear = lambda n: O(n, (n, oo))
quadratic = lambda n: O(n**2, (n,oo))
logarithmic = lambda n: O(sympy.log(n, 2), (n,oo))
exponential = lambda n: O(2**n, (n,oo))
constant = lambda _: O(1)

class Complexity(pydrogen.Typical):
    """ Complexity approximation for a small subset of Python. """

    # store the complexity of some known functions for added expressibility
    # technically, range in python 3 has constant complexity since it returns a
    # generator, but within the context of loops we say it is linear
    functions = {'len': constant, 'print': constant, 'range': linear}

    def preprocess(self, context):
        syms = []
        # allow for specification of running time of certain functions as well (?)
        if 'functions' in context and type(context['functions']) == dict:
            self.functions.update(context['functions'])
            del context['functions']
        for var, symbol in context.items():
            if type(symbol) == str:
                symbol = symbols(symbol)
                context[var] = symbol
            syms.append((symbol, oo))

    def Statements(self, ss, context=None): return sum(ss.post(context)[0])
    def Assign(self, targets, e, context=None): return constant(None) + e.post(context)
    # if iterable maps to a symbol, use it. otherwise, assume it is constant.
    def For(self, target, itr, ss, orelse, context=None):
        # we make the simplifying assumption that whatever being iterated over
        # is either:
        #   a named variable, where we look up the context for a symbol and
        #       regard as constant size if not present
        #   a built-in sequence (list, string, dictionary, tuple, set) which we
        #       regard as constant size
        #   a function call, where we assume that the complexity of the function
        #       call is also the size of the iterable returned. This is
        #       obviously not always the case but we assume relatively
        #       simple functions.
        if hasattr(itr.pre(), 'id'):
            itr = itr.pre().id
            if itr in context:
                loops = linear(context[itr])
            else:
                loops = constant(None)
        elif type(itr.pre()) in (ast.List, ast.Tuple, ast.Str, ast.Dict, ast.Set):
            loops = constant(None)
        else:
            loops = itr.post(context)
        work = ss.post(context)
        # sympy raises error if you try to do O(1) * O(x, (x,oo)) for e.g.
        return loops * work if work.contains(loops) else work * loops
    def BoolOp(self, es, context=None): return constant(None) + sum(es.post(context))
    def BinOp(self, e1, e2, context): return constant(None) + e1.post(context) + e2.post(context)
    def Call(self, func, args, context=None):
        func = func.pre()
        if context is None:
            context = {}
        if func.id in self.functions:
            symbol = None
            # check context if any arg maps to a symbol; take first one we find
            # and return complexity class of the function.
            for arg in args.pre():
                if arg.id in context:
                    symbol = context[arg.id]
                    break
            if symbol or self.functions[func.id] == constant:
                return self.functions[func.id](symbol)
        # otherwise, attempt to interpret the function directly
        for _, module in sys.modules.items():
            scope = module.__dict__
            if func.id in scope and callable(scope[func.id]):
                func = scope[func.id]
                return Complexity(func, **context, functions=self.functions).Complexity
        raise pydrogen.PydrogenError("failed to interpret {}".format(func.id))
    def Num(self, n, context=None): return 0
    def NameConstant(self): return 0
    def Name(self, id, context=None): return 0
    def If(self, test, body, orelse, context=None):
        return test.post(context) + body.post(context) + orelse.post(context)
    def Compare(self, l, r, context=None):
        return constant(None) + l.post(context) + r.post(context)

@Complexity()
def average(items):
    total = 0
    for item in [1,2,3,4]:
        for item in items:
            total = total + item
    return total/len(items)

print('The asymptotic complexity of average() is ' + str(average.Complexity) + '.')