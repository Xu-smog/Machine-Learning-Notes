import torch
from torch import nn
'''
x = torch.rand(100, 16, 784)

layer = nn.BatchNorm1d(16)
out = layer(x)

print(layer.running_mean)
print(layer.running_var)
'''
x = torch.randn(1, 16, 7, 7)

layer = nn.BatchNorm2d(16)
out = layer(x)

print(layer.weight)
print(layer.weight.shape)
print(layer.bias.shape)
print(vars(layer))