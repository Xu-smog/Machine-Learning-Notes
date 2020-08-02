import torch
import numpy as np

#view/rshape
a = torch.rand(4, 1, 28, 28)
print(a.shape)
print(a.view(4, 28*28))
print(a.view(4, 28*28).shape)
print(a.view(4*28, 28).shape)
print(a.view(4*1, 28, 28).shape)

#squeeze/unsqueeze
print(a.shape)
print(a.unsqueeze(0).shape)
print(a.unsqueeze(-1).shape)
print(a.unsqueeze(4).shape)
print(a.unsqueeze(-4).shape)
print(a.unsqueeze(-5).shape)

a = torch.tensor([1.2, 2.3])
print(a.unsqueeze(-1))
print(a.unsqueeze(0))

b = torch.rand(32)
f = torch.rand(4, 32, 14, 14)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape)

a = torch.rand(1, 32, 1, 2)
print(a.squeeze().shape)
print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(-1).shape)
print(b.squeeze(1).shape)
print(b.squeeze(-4).shape)

#expand/repeat
print(b.expand(4, 32, 14, 14).shape)
print(b.expand(-1, 32, 14, -1).shape)

print(b.repeat(4, 32, 1, 1).shape)
print(b.repeat(4, 1, 1, 1).shape)
print(b.repeat(4, 1, 32, 32).shape)

#.t
a = torch.randn(3, 4)
print(a.t())

#transpose
b = torch.rand(4, 3, 28, 32)
print(b.transpose(1, 3).shape)
print(b.transpose(1, 3).transpose(1, 2).shape)

#permute
print(b.permute(0, 2, 3, 1).shape)