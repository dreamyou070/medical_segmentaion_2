from torch import nn
import torch

alpha = nn.Parameter(torch.ones(1))
print(alpha)
alpha = alpha.to('cpu')
print(alpha)