# https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.autodecoder import AutoencoderKL

#x16_out = torch.randn(1, 1280, 16, 16)
#x32_out = torch.randn(1, 640, 32, 32)
#x64_out = torch.randn(1, 320, 64, 64)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_batchnorm = True,
                 use_instance_norm = True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                         nn.LayerNorm([mid_channels, int(20480 / mid_channels), int(20480 / mid_channels)]),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                         nn.LayerNorm([mid_channels, int(20480 / mid_channels), int(20480 / mid_channels)]),
                                         nn.ReLU(inplace=True))

        if use_batchnorm :
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(mid_channels),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nn.ReLU(inplace=True))
        if use_instance_norm :
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.InstanceNorm2d(mid_channels),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.InstanceNorm2d(out_channels),
                                             nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_batchnorm = True,
                 use_instance_norm = True) :
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,
                                   use_batchnorm = use_batchnorm,
                                   use_instance_norm = use_instance_norm)
                                  # norm_type=norm_type) # This use batchnorm
        else: # Here
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm = use_batchnorm,
                                   use_instance_norm = use_instance_norm)
    def forward(self, x1, x2):

        # [1] x1
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # [2] concat
        x = torch.cat([x2, x1], dim=1) # concatenation
        # [3] out conv
        x = self.conv(x)
        return x

class Up_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels = in_channels,
                                     out_channels = out_channels,
                                     kernel_size=kernel_size,
                                     stride=kernel_size)
    def forward(self, x1):
        x = self.up(x1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SemanticModel(nn.Module):
    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_layer_norm=True,
                 use_instance_norm=True,
                 mask_res=128,
                 high_latent_feature=False,
                 init_latent_p=1):
        super(SemanticModel, self).__init__()

        c = 320

        self.mlp_layer_1 = torch.nn.Linear(1280, c)
        self.upsample_layer_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.mlp_layer_2 = torch.nn.Linear(640, c)
        self.upsample_layer_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.mlp_layer_3 = torch.nn.Linear(320, c)
        self.upsample_layer_3 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)

        self.use_layer_norm = use_layer_norm
        self.use_instance_norm = use_instance_norm
        if self.use_layer_norm:
            if mask_res == 128:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160 , 128, 128]),)
            if mask_res == 256:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160*3, kernel_size=2),
                                                       nn.LayerNorm([160*3, 128, 128]),
                                                       Up_conv(in_channels=160*3, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160, 256,256]))
            if mask_res == 512:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160*3, kernel_size=2),
                                                       nn.LayerNorm([160*3, 128, 128]),
                                                       Up_conv(in_channels=160*3, out_channels=160*2, kernel_size=2),
                                                       nn.LayerNorm([160*2, 256, 256]),
                                                       Up_conv(in_channels=160*2, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160, 512, 512]))
        elif self.use_instance_norm :
            if mask_res == 128:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160, kernel_size=2),
                                                       nn.InstanceNorm2d(160),)
            if mask_res == 256:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160*3, kernel_size=2),
                                                       nn.InstanceNorm2d(160*3),
                                                       Up_conv(in_channels=160*3, out_channels=160, kernel_size=2),
                                                       nn.InstanceNorm2d(160))
            if mask_res == 512:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160*3, kernel_size=2),
                                                       nn.InstanceNorm2d(160*3),
                                                       Up_conv(in_channels=160*3, out_channels=160*2, kernel_size=2),
                                                       nn.InstanceNorm2d(160*2),
                                                       Up_conv(in_channels=160*2, out_channels=160, kernel_size=2),
                                                       nn.InstanceNorm2d(160))
        else :
            if mask_res == 128:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160, kernel_size=2),
                                                       )
            if mask_res == 256:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160*3, kernel_size=2),
                                                       Up_conv(in_channels=160*3, out_channels=160, kernel_size=2),)
            if mask_res == 512:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160*3, kernel_size=2),
                                                       Up_conv(in_channels=160*3, out_channels=160*2, kernel_size=2),
                                                       Up_conv(in_channels=160*2, out_channels=160, kernel_size=2),)
        self.outc = OutConv(160, n_classes)

    def dim_and_res_up(self, mlp_layer, upsample_layer, x):
        # [batch, dim, res, res] -> [batch, res*res, dim]
        batch, dim, res, res = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(1, res * res, dim)
        # [1] dim change
        x = mlp_layer(x)  # x = [batch, res*res, new_dim]
        new_dim = x.shape[-1]
        x = x.permute(0, 2, 1).contiguous().reshape(1, new_dim, res, res)
        # [2] res change
        x = upsample_layer(x)
        return x

    def forward(self, x16_out, x32_out, x64_out):

        x16_out = self.dim_and_res_up(self.mlp_layer_1, self.upsample_layer_1, x16_out)  # [batch, 320, 64,64]
        x32_out = self.dim_and_res_up(self.mlp_layer_2, self.upsample_layer_2, x32_out)  # [batch, 320, 64,64]
        x64_out = self.dim_and_res_up(self.mlp_layer_3, self.upsample_layer_3, x64_out)  # [batch, 320, 64,64]
        x = torch.cat([x16_out, x32_out, x64_out], dim=1)  # [batch, 960, res,res]
        # non linear model ?
        x = self.segmentation_head(x) # [batch, 160, 256,256]
        logits = self.outc(x)  # 1, 4, 128,128
        return logits
