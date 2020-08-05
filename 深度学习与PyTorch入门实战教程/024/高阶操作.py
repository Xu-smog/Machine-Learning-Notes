import torch

#where
cond = torch.rand(2, 2)
print(cond)
a = torch.zeros_like(cond)
b = torch.ones_like(cond)
print(a)
print(b) 
print(torch.where(cond > 0.5, a, b))

#gather
prob = torch.randn(4, 10)
idx = prob.topk(dim=1, k=3)
print(idx)
print(idx[1])
idx = idx[1]

label = torch.arange(10) + 100
print(torch.gather(label.expand(4, 10), dim=1, index=idx.long()))