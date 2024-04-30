from torch import nn
import torch
channel = 320

map = nn.Conv2d(channel, 1, 11, 1, 5)
input = torch.randn(1,320, 25,25)
# upscaling ... ?

output = map(input)
print(output.size())