import torch
from torch import nn

x4 = torch.randn((1, 512, 8, 8))
#x4 = x4.permute(0, 2, 3, 1) # 1,8,8,512
#x4 = x4.reshape(1, -1, 512) # 1,64,512

x3 = torch.randn((1, 320, 16, 16))
x3 = x3.permute(0, 2, 3, 1)
x3 = x3.reshape(1, -1, 320) # 1,256,320
# up scaling
#up = nn.ConvTranspose2d(512, 312, kernel_size=2, stride=2) #
up = nn.Sequential(nn.Conv2d(512,320, 7, 1, 3),
                             nn.BatchNorm2d(320),
                             nn.Sigmoid(),
                             nn.UpsamplingBilinear2d(scale_factor=2))
x4_up = up(x4)

# upsacling
print(f'x4 shape = {x4.shape}')
print(f'x3 shape = {x3.shape}')
print(f'x4_up shape = {x4_up.shape}')