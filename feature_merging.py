import torch
import torch.nn as nn
k1, s1, p1 = 7,4,3
k2, s2, p2 = 3,2,1

conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                  kernel_size=k1, stride=s1, padding=p1)
conv2 = nn.Conv2d(in_channels=3, out_channels=128,
                    kernel_size=k2, stride=s2, padding=p2)

input = torch.randn(1, 3, 512, 512)
output_1 = conv1(input) # 1/4
print(f'output_1 shape: {output_1.shape}')
output_2 = conv2(input) # 1/2
print(f'output_2 shape: {output_2.shape}')