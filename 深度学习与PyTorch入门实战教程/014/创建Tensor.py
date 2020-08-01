import torch
import numpy as np

#数据导入
a = np.array([2, 3.3])
print(torch.from_numpy(a))

a = np.ones([2, 3])
print(torch.from_numpy(a))

print(torch.tensor([2., 3.2]))
print(torch.FloatTensor([2., 3.2]))
print(torch.tensor([[2., 3.2], [1., 22.3]]))

#初始化
print(torch.empty(1))
print(torch.Tensor(2, 3))
print(torch.IntTensor(2, 3))
print(torch.FloatTensor(2, 3))

#默认类型
print(torch.tensor([1.2, 3]).type())
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2, 3]).type())

#随机初始化
#均匀分布
print(torch.rand(3, 3))
a = torch.rand(3, 3)
print(torch.rand_like(a))
print(torch.randint(1, 10, [3, 3]))

#正态分布
print(torch.randn(3, 3))
print(torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1)))

#full
print(torch.full([2, 3], 7))
print(torch.full([], 7))
print(torch.full([1], 7))

#arange
print(torch.arange(0, 10))
print(torch.arange(0, 10, 2))

#linspace/logspace
print(torch.linspace(0, 10, steps=4))
print(torch.linspace(0, 10, steps=10))
print(torch.linspace(0, 10, steps=11))
print(torch.logspace(0, -1, steps=10))
print(torch.logspace(0, 1, steps=10))

#ones/zeros/eye
print(torch.ones(3, 3))
print(torch.zeros(3, 3))
print(torch.eye(3, 4))
print(torch.eye(3))
a = torch.zeros(3, 3)
print(torch.ones_like(a))

#randperm
print(torch.randperm(10))
a = torch.rand(2, 3)
b = torch.rand(2, 2)
print(a)
print(b)
idx = torch.randperm(2)
print(idx)
print(a[idx])
print(b[idx])
