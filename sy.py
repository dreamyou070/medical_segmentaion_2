import torch
from torch import nn

x1 = torch.randn((1, 64, 64, 64))
x2 = torch.randn((1, 128, 32, 32))
x3 = torch.randn((1, 320, 16, 16))
x4 = torch.randn((1, 512, 8, 8))

##
feature_up_4_3 = nn.Sequential(nn.Conv2d(512, 320, 7, 1, 3),
                               nn.BatchNorm2d(320),
                               nn.UpsamplingBilinear2d(scale_factor=2))
feature_up_3_2 = nn.Sequential(nn.Conv2d(320, 128, 7, 1, 3),
                               nn.BatchNorm2d(128),
                               nn.UpsamplingBilinear2d(scale_factor=2))
feature_up_2_1 = nn.Sequential(nn.Conv2d(128, 64, 7, 1, 3),
                               nn.BatchNorm2d(64),
                               nn.UpsamplingBilinear2d(scale_factor=2))
x43 = feature_up_4_3(x4)
x4_out = torch.cat([x43, x3], dim=1)
x32 = feature_up_3_2(x3)
x3_out = torch.cat([x3_out, x2], dim=1)
x21 = feature_up_2_1(x2)
x2_out = torch.cat([x2_out, x1], dim=1)
# (3,3) convolution
feature_4_conv = nn.Sequential(nn.Conv2d(640, 320, kernel_size=3, stride=1, padding=1, dilation=1),
                               nn.Sigmoid())
feature_3_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                                 nn.Sigmoid())
feature_2_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                                 nn.Sigmoid())
print(x4_out.shape)
print(x3_out.shape)
print(x2_out.shape)
x4_out = feature_4_conv(x4_out)
x3_out = feature_3_conv(x3_out)
x2_out = feature_2_conv(x2_out)
print(x4_out.shape)
print(x3_out.shape)
print(x2_out.shape)