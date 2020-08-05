import torch

#norm
a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(b)
print(c)

print(a.norm(1), b.norm(1), c.norm(1))
print(a.norm(2), b.norm(2), c.norm(2))
print(b.norm(1, dim=1))
print(b.norm(2, dim=1))
print(c.norm(1, dim=0))
print(c.norm(2, dim=0))

#mean/sum/min/max/prod argmax返回下标
a = torch.arange(8).view(2, 4).float()
print(a)
print(a.min(), a.max(), a.mean(), a.prod())
print(a.sum())
print(a.argmax(), a.argmin())

a = torch.arange(0, 16, 2).view(2, 4).float()
print(a)
print(a.argmax(), a.argmin())

a = torch.randn(4, 10)
print(a.argmax())
print(a.argmax(dim=1))

#dim/keepdim
print(a.max(dim=1))
print(a.argmax(dim=1))
print(a.max(dim=1, keepdim=True))
print(a.argmax(dim=1, keepdim=True))

#top-k/k-th
print(a.topk(3, dim=1))
print(a.topk(3, dim=1, largest=False))
print(a.kthvalue(8, dim=1))
print(a.kthvalue(3))
print(a.kthvalue(3, dim=1))

#compare
print(a > 0)
print(torch.gt(a, 0))
print(a != 0)

a = torch.ones(2, 3)
b = torch.randn(2, 3)
print(torch.eq(a, a))
print(torch.equal(a, a))