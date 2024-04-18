from torch import nn
import torch


l2_loss = nn.MSELoss(reduction='none')
a = torch.randn(1,4,64,64)
b = torch.randn(1,4,64,64)
loss = l2_loss(a,b)
print(loss.shape)