from scipy.optimize import minimize, rosen, rosen_der
import torch
from torch.autograd import Variable

def var(i, requires_grad=False):
	return Variable(torch.tensor(i, dtype=torch.double), requires_grad=requires_grad)

def list2tensorl(x_list):
	value_list = list()
	print(x_list)
	for i in x_list:
		print(var(i))
		value_list.append(var(i))
	return value_list

def f(x):
	print('x', x)
	t = torch.square(x[0].sub(var(1.0))).add(torch.square(x[1].sub(var(2.5))))
	print('t', t)
	return t

# fun = lambda x: f(x)# f(list2tensorl(x)).data.item()
# bnds = [(0.1, 1.2), (0.3, 4.0)]
# # bnds = ((var(0.1).item(), var(1.2).item(), (var(0.3).item(), var(4.0).item())))
# # res = minimize(fun, (var(1).item(),var(3).item()), method='SLSQP', bounds=bnds)

# print(res)
# x = [i for i in res.x]
# print(x)

# print(var(0.0).item())

Theta = var(3.0, requires_grad=True)

def p(theta, x):
	return theta.mul(theta).add(theta.mul(x)).sub(torch.exp(theta.mul(x)))


x1 = var(2.0)*Theta.sub(var(6.0))
x2 = var(0.0)

f1 = p(Theta, x1)
f2 = p(Theta, x2)

d1 = torch.autograd.grad(f1, Theta, retain_graph=True)
d2 = torch.autograd.grad(f2, Theta, retain_graph=True)

print('derivative1', d1)
print('derivative2', d2)




