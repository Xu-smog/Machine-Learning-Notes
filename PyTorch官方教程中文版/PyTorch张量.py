import torch

'''
x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)
print(x[0][0])
print(type(x[0][0]))

x = torch.tensor([5.5, 3])
print(x)

x = torch.rand(5, 3)
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

x = torch.rand(5, 3)
y = torch.rand(5, 3)
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

x = torch.rand(5, 3)
print(x)
print(x[:, 1])
print(x[:, 2])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
'''
x = torch.randn(1)
print(x)
print(x.item())
