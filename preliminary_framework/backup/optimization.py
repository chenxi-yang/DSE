
import nlopt
from numpy import *



def myfunc(x, grad):
    # print(x)
    return x[0]*x[0]*x[0]


def optimization(myfunc):
    opt = nlopt.opt(nlopt.GN_DIRECT, 1)
    opt.set_lower_bounds([-10])
    opt.set_upper_bounds([10])
    opt.set_min_objective(myfunc)
    opt.set_stopval(0.0)
    opt.set_maxeval(1000)
    x = opt.optimize([0.5])
    minf = opt.last_optimum_value()

    print('opt, ', x[0])
    print('f=', minf)

optimization(myfunc)