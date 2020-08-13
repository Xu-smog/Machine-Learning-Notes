import torch
from torch.nn import functional as F

x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x @ w.t()
print(logits.shape)

pred = F.softmax(logits, dim=1)
print(pred.shape)

pred_log = torch.log(pred)

print(F.cross_entropy(logits, torch.tensor([3])))

print(F.nll_loss(pred_log, torch.tensor([3])))