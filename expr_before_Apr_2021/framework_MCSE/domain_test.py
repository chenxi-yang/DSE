from helper import *
from constants import *
import domain

# x0 = domain.Interval(var(-1.0).mul(PI_HALF), PI_HALF)
x1 = domain.Interval(var(0.0), PI)
x2 = domain.Interval(PI_HALF, PI)
x3 = domain.Interval(PI_HALF, PI_TWICE)
x4 = domain.Interval(var(0.0), var(2.8))
xx4 = var(1.7256)

# y0 = x0.cos()
y1 = x1.cos()
y2 = x2.cos()
y3 = x3.cos()
y4 = x4.cos()
y4_sin = x4.sin()
yy4 = torch.sin(xx4)

# print('y0', y0.left, y0.right)
print('y1', y1.left, y1.right) # y
print('y2', y2.left, y2.right) # y
print('y3', y3.left, y3.right) # y
print('y4', y4.left, y4.right) # no
print('y4_sin', y4_sin.left, y4_sin.right)
print('yy4', yy4) #


# x_mod_1 = domain.Interval(var(3.0), var(5.3))
# x_mod_2 = domain.Interval(var(2.0), var(7.0))
# y_mod = x_mod_1.fmod(x_mod_2)

# # print('y_mod', y_mod.left, y_mod.right) # y

# x_neg = domain.Interval(var(-3.0).mul(PI_HALF), var(5.0).mul(PI_TWICE))
# y_neg = x_neg.cos()
# print('y_neg', y_neg.left, y_neg.right)