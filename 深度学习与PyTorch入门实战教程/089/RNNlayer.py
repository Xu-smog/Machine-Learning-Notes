import torch
from torch import nn
'''
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
print(rnn)
x = torch.randn(10, 3, 100)
out, h = rnn(x, torch.zeros(1, 3, 20))
print(out.shape, h.shape)

rnn = nn.RNN(100, 10, num_layers=2)
print(rnn._parameters.keys())
print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)
print(rnn.weight_hh_l1.shape, rnn.weight_ih_l1.shape)
'''
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
print(rnn)
x = torch.randn(10, 3, 100)
out, h = rnn(x)
print(out.shape, h.shape)
