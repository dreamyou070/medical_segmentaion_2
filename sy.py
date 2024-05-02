from torch import nn
import torch
a = nn.parameter.Parameter(data=torch.tensor([1.0]), requires_grad=True)
print(a)