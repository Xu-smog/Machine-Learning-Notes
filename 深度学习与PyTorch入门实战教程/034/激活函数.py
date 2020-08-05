import torch
from torch.nn import functional as F

a = torch.linspace(-1, 1, 10)
print(a)

#sigmoid
print(torch.sigmoid(a))

#tanh
print(torch.tanh(a))

#relu
print(torch.relu(a))
print(F.relu(a))
