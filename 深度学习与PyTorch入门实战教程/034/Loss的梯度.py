import torch
from torch.nn import functional as F

'''
x = torch.ones(1)
w = torch.full([1], 2)

w.requires_grad_()
mse = F.mse_loss(x*w, torch.ones(1))
print(x, w, mse)

#autograd.grad
#print(torch.autograd.grad(mse, [w]))

#loss.backward
mse.backward()
print(w.grad)
'''

a = torch.rand(3)
a.requires_grad_()

p = F.softmax(a, dim=0)

#autograd.grad
print(torch.autograd.grad(p[1], [a]))