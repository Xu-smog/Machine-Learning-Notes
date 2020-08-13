import torch
from torch import nn
from torch.nn import functional as F 

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(32*5*5, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz, 32*5*5)
        logits = self.fc_unit(x)
        return logits
        