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
                                         nn.LayerNorm(
                                             [mid_channels, int(20480 / mid_channels), int(20480 / mid_channels)]),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                         nn.LayerNorm(
                                             [mid_channels, int(20480 / mid_channels), int(20480 / mid_channels)]),
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

class SemanticSeg(nn.Module):
    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_batchnorm=True,
                 use_instance_norm = True,
                 mask_res = 128,
                 high_latent_feature = False):

        super(SemanticSeg, self).__init__()

        factor = 2 if bilinear else 1
        self.up1 = Up(1280, 640 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm, use_instance_norm)

        if mask_res == 128:
            self.segmentation_head = nn.Sequential(Up_conv(in_channels = 320, out_channels = 160, kernel_size=2))
        if mask_res == 256 :
            self.segmentation_head = nn.Sequential(Up_conv(in_channels = 320, out_channels = 160, kernel_size=2),
                                                   Up_conv(in_channels = 160, out_channels = 160, kernel_size=2))
        if mask_res == 512 :
            self.segmentation_head = nn.Sequential(Up_conv(in_channels=320, out_channels=160, kernel_size=2),
                                                   Up_conv(in_channels=160, out_channels=160, kernel_size=2),
                                                   Up_conv(in_channels=160, out_channels=160, kernel_size=2))
        self.high_latent_feature = high_latent_feature
        if self.high_latent_feature :
            self.feature_generator = nn.Sequential(nn.Sigmoid())
        else :
            self.feature_generator = nn.Sequential(nn.Conv2d(320, 4, kernel_size=3, padding=1),
                                                   nn.Sigmoid())
        self.outc = OutConv(160, n_classes)


    def forward(self, x16_out, x32_out, x64_out):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)       # 1,320,64,64
        gen_feature = self.feature_generator(x) # 1, 4, 64, 64
        x = self.segmentation_head(x)
        logits = self.outc(x)  # 1, 4, 128,128
        return gen_feature, logits

class SemanticSeg_Gen(nn.Module):
    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_batchnorm=True,
                 use_instance_norm = True,
                 mask_res = 128,
                 high_latent_feature = False):

        super(SemanticSeg, self).__init__()

        factor = 2 if bilinear else 1
        self.up1 = Up(1280, 640 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm, use_instance_norm)

        if mask_res == 128:
            self.segmentation_head = nn.Sequential(Up_conv(in_channels = 320, out_channels = 160, kernel_size=2))
        if mask_res == 256 :
            self.segmentation_head = nn.Sequential(Up_conv(in_channels = 320, out_channels = 160, kernel_size=2),
                                                   Up_conv(in_channels = 160, out_channels = 160, kernel_size=2))
        if mask_res == 512 :
            self.segmentation_head = nn.Sequential(Up_conv(in_channels=320, out_channels=160, kernel_size=2),
                                                   Up_conv(in_channels=160, out_channels=160, kernel_size=2),
                                                   Up_conv(in_channels=160, out_channels=160, kernel_size=2))
        self.high_latent_feature = high_latent_feature
        if self.high_latent_feature :
            self.feature_generator = nn.Sequential(nn.Sigmoid())
        else :
            self.feature_generator = nn.Sequential(nn.Conv2d(320, 4, kernel_size=3, padding=1),
                                                   nn.Sigmoid())
        self.outc = OutConv(160, n_classes)

        latent_dim = 320
        if not self.high_latent_feature:
            latent_dim = 4
        self.decoder_model = AutoencoderKL(spatial_dims=2,
                                          out_channels=3,
                                          num_res_blocks=(2, 2, 2, 2),
                                          num_channels=(32, 64, 64, 64),
                                          attention_levels=(False, False, True, True),
                                          latent_channels=latent_dim,
                                          norm_num_groups=32,
                                          norm_eps=1e-6,
                                          with_encoder_nonlocal_attn=True,
                                          with_decoder_nonlocal_attn=True,
                                          use_flash_attention=False,
                                          use_checkpointing=False,
                                          use_convtranspose=False)


    def reconstruction(self, x):
        return self.decoder_model(x)

    def forward(self, x16_out, x32_out, x64_out):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)       # 1,320,64,64
        gen_feature = self.feature_generator(x) # 1, 4, 64, 64
        # [1] recon
        reconstruction, z_mu, z_sigma = self.reconstruction(gen_feature) # 1,3,512,512

        x = self.segmentation_head(x)
        logits = self.outc(x)  # 1, 4, 128,128
        #return gen_feature, logits
        return reconstruction, z_mu, z_sigma, logits
