import torch
import numpy as np

a = torch.randn(2, 3)

print(a.type())
#torch.FloatTensor
print(type(a))
#<class 'torch.Tensor'>
print(isinstance(a, torch.FloatTensor))
#True
print(type(isinstance(a, torch.FloatTensor)))
#<class 'bool'>

print(isinstance(a, torch.cuda.FloatTensor))
#False
a = a.cuda()
print(isinstance(a, torch.cuda.FloatTensor))
#True

print(torch.tensor(1.))
#tensor(1.)
a = torch.tensor(1.3)
print(a)
#tensor(1.3000)
print(a.shape)
#torch.Size([])
print(len(a.shape))
#0
print(a.dim())
#0
print(a.size())
#torch.Size([])

print(torch.tensor([1.1]))
#tensor([1.1000])
print(torch.tensor([1.1, 2.2]))
#tensor([1.1000, 2.2000])
print(torch.FloatTensor(1))
#tensor([0.])
print(torch.FloatTensor(2))
#tensor([-1.0842e-19,  1.8875e+00])
a = np.ones(2)
print(a)
#[1. 1.]
print(torch.from_numpy(a))
#tensor([1., 1.], dtype=torch.float64)

a = torch.ones(2)
print(a.shape)
#torch.Size([2])

a = torch.randn(2, 3)
print(a)
#tensor([[-0.5266, -0.2352, -0.2069],
#        [-0.3682,  0.5999, -0.1665]])
print(a.shape)
#torch.Size([2, 3])
print(a.size(0))
#2
print(a.size(1))
#3
print(a.shape[1])
#3

a = torch.rand(1, 2, 3)
print(a)
#tensor([[[0.8217, 0.0034, 0.0933],
#         [0.4304, 0.6868, 0.0742]]])
print(a.shape)
#torch.Size([1, 2, 3])
print(a[0])
#tensor([[0.8217, 0.0034, 0.0933],
#        [0.4304, 0.6868, 0.0742]])
print(list(a.shape))
#[1, 2, 3]

a = torch.randn(2, 3, 4, 5)
print(a)
#tensor([[[[ 0.5499, -0.6092, -0.0719,  0.1592,  0.7542],        
#          [ 0.4364,  0.3917, -0.0401, -3.4411,  0.9238],        
#          [ 0.1338, -1.5961,  0.5761, -0.8205,  0.0909],        
#          [ 1.0058,  0.2932,  0.0335, -0.3985,  0.9532]],       
#           ......
#
#           ......
#         [[ 0.5785, -0.6138,  1.2770, -0.0637, -1.5136],        
#          [ 0.6455, -0.5873, -0.2975,  0.8403, -0.1647],        
#          [-0.1051,  0.3338,  0.0982,  1.2400, -3.0205],        
#          [ 0.4300, -1.0216, -0.8845, -1.2321,  0.5518]]]])  
print(a.shape)
#torch.Size([2, 3, 4, 5])
print(a.numel())
#120
print(a.dim())
#4