# https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                     kernel_size=kernel_size, stride=kernel_size)
    def forward(self, x1):
        x = self.up(x1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Segmentation_Head_a(nn.Module):

    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_batchnorm=True,
                 use_instance_norm = True,
                 mask_res = 128,
                 use_init_query = False,):
        super(Segmentation_Head_a, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.use_init_query = use_init_query
        if self.use_init_query :
            self.init_conv = nn.Conv2d(4, 320, kernel_size=3, padding=1, bias=False)
            self.double_conv = DoubleConv(640, 320, use_batchnorm = use_batchnorm, use_instance_norm = use_instance_norm)

        self.up1 = Up(1280, 640 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm, use_instance_norm)

        if self.mask_res == 64 :
            self.up3 = nn.Conv2d(320, 160, kernel_size=3, padding=1, bias=False)
        if self.mask_res == 128:
            self.up3 = Up_conv(in_channels = 320,
                                out_channels = 160,
                                kernel_size=2) # 64 -> 128 , channel 320 -> 160
        if self.mask_res == 256 :
            self.up4 = Up_conv(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2)  # 128 -> 256
        if self.mask_res == 512 :
            self.up4 = Up_conv(in_channels=160,
                               out_channels=160,
                               kernel_size=2)  # 128 -> 256
            self.up5 = Up_conv(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2)
        self.outc = OutConv(160, n_classes)
    def forward(self, x16_out, x32_out, x64_out,
                x_init = None):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)    # 1,320,64,64
        if self.use_init_query :
            x_init = self.init_conv(x_init)   # 1, 320, 64, 64
            x = torch.cat([x, x_init], dim=1) # 1, 640, 64, 64
            x = self.double_conv(x)           # 1, 320, 64, 64
        x3_out = self.up3(x)  # 1,160,128,128
        x_in = x3_out
        if self.mask_res == 256 :
            x4_out = self.up4(x3_out)
            x_in = x4_out
        if self.mask_res == 512 :
            x4_out = self.up4(x3_out)
            x5_out = self.up5(x4_out)
            x_in = x5_out
        logits = self.outc(x_in)  # 1,4, 128,128
        return logits

class Segmentation_Head_b(nn.Module):

    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_batchnorm=True,
                 use_instance_norm=True,
                 mask_res=128,
                 use_init_query=False):
        super(Segmentation_Head_b, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up(1280, 640 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up3 = Up(640, 320 // factor, bilinear, use_batchnorm, use_instance_norm)
        if self.mask_res == 64 :
            self.up4 = nn.Conv2d(320, 160, kernel_size=3, padding=1, bias=False)
        elif self.mask_res == 128 :
            self.up4 = Up_conv(in_channels = 320,
                                out_channels=160,
                                kernel_size=2)
        elif self.mask_res == 256 :
            self.up5 = Up_conv(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2)
        elif self.mask_res == 512:
            self.up5 = Up_conv(in_channels=160,
                               out_channels=160,
                               kernel_size=2)
            self.up6 = Up_conv(in_channels=160,
                               out_channels=160,
                               kernel_size=2)
        self.outc = (OutConv(160, n_classes))

    def forward(self, x16_out, x32_out, x64_out):

        x1_out = self.up1(x16_out, x32_out)  # 1,640,32,32
        x2_out = self.up2(x32_out, x64_out)  # 1,320,64,64
        x3_out = self.up3(x1_out, x2_out)    # 1,320,64,64
        x4_out = self.up4(x3_out)            # 1,160, 256,256
        x_in = x4_out
        if self.mask_res == 256 :
            x5_out = self.up5(x4_out)        # 1,160,256,256
            x_in = x5_out
        elif self.mask_res == 512 :
            x5_out = self.up5(x4_out)
            x6_out = self.up6(x5_out)
            x_in = x6_out
        logits = self.outc(x_in)  # 1,3,256,256
        return logits

class Segmentation_Head_c(nn.Module):

    def __init__(self,
                 n_classes, bilinear=False,
                 use_batchnorm=True,
                 mask_res = 128,
                 norm_type = 'batch_norm',
                 use_instance_norm = True,
                 use_init_query = False,
                 attn_factor = 5):
        super(Segmentation_Head_c, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1 # always 1
        self.up1 = Up(1280, 640 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up3 = Up(640, 320 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up4 = Up_conv(in_channels = 640,
                            out_channels = 160,
                            kernel_size=2)
        if self.mask_res == 256 :
            self.up5 = Up_conv(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2)
        elif self.mask_res == 512 :

            self.up5 = Up_conv(in_channels=160,
                               out_channels=160,
                               kernel_size=2)
            self.up6 = Up_conv(in_channels=160,
                               out_channels=160,
                               kernel_size=2)
        self.outc = OutConv(160, n_classes)

    def forward(self, x16_out, x32_out, x64_out):

        x1_out = self.up1(x16_out, x32_out)     # 1,640,32,32
        x2_out = self.up2(x32_out, x64_out)     # 1,320,64,64
        x3_out = self.up3(x1_out, x2_out)       # 1,320,64,64
        x = torch.cat([x3_out, x64_out], dim=1) # 1,640,64,64
        x4_out = self.up4(x)                    # 1,320,128,128
        x_in = x4_out
        if self.mask_res == 256 :
            x5_out = self.up5(x4_out)            # 1,320,256,256
            x_in = x5_out
        elif self.mask_res == 512 :
            x5_out = self.up5(x4_out)
            x6_out = self.up6(x5_out)
            x_in = x6_out

        logits = self.outc(x_in)  # 1,3,256,256
        return x_in, logits

class Segmentation_Head_d(nn.Module):

    def __init__(self,
                 n_classes, bilinear=False,
                 use_batchnorm=True,
                 mask_res = 128,
                 norm_type = 'batch_norm',
                 use_instance_norm = True,
                 use_init_query = False,
                 attn_factor = 5):
        super(Segmentation_Head_d, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1 # always 1
        self.up1 = Up(1280*attn_factor, 640*attn_factor // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up2 = Up(640*attn_factor, 320*attn_factor // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up3 = Up(640*attn_factor, 320*attn_factor // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up4 = Up_conv(in_channels = 640*attn_factor,
                            out_channels = 160,
                            kernel_size=2)
        if self.mask_res == 256 :
            self.up5 = Up_conv(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2)
        elif self.mask_res == 512 :
            self.up5 = Up_conv(in_channels=160,
                               out_channels=160,
                               kernel_size=2)
            self.up6 = Up_conv(in_channels=160,
                               out_channels=160,
                               kernel_size=2)
        self.outc = OutConv(160, n_classes)

    def forward(self, x16_out, x32_out, x64_out):

        x1_out = self.up1(x16_out, x32_out)     # 1,640,32,32
        x2_out = self.up2(x32_out, x64_out)     # 1,320,64,64
        x3_out = self.up3(x1_out, x2_out)       # 1,320,64,64
        x = torch.cat([x3_out, x64_out], dim=1) # 1,640,64,64
        x4_out = self.up4(x)                    # 1,320,128,128
        x_in = x4_out
        if self.mask_res == 256 :
            x5_out = self.up5(x4_out)            # 1,320,256,256
            x_in = x5_out
        elif self.mask_res == 512 :
            x5_out = self.up5(x4_out)
            x6_out = self.up6(x5_out)
            x_in = x6_out
        logits = self.outc(x_in)  # 1,3,256,256
        return

class Segmentation_Attn_Head_A(nn.Module):

    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_batchnorm=True,
                 use_instance_norm = True,
                 mask_res = 128,
                 use_init_query = False,):
        super(Segmentation_Head_a, self).__init__()

        self.n_classes = n_classes
        self.mask_res = mask_res
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.use_init_query = use_init_query
        if self.use_init_query :
            self.init_conv = nn.Conv2d(4, 320, kernel_size=3, padding=1, bias=False)
            self.double_conv = DoubleConv(640, 320, use_batchnorm = use_batchnorm, use_instance_norm = use_instance_norm)

        self.up1 = Up(1, 640 // factor, bilinear, use_batchnorm, use_instance_norm)
        self.up2 = Up(640, 320 // factor, bilinear, use_batchnorm, use_instance_norm)

        if self.mask_res == 64 :
            self.up3 = nn.Conv2d(320, 160, kernel_size=3, padding=1, bias=False)
        if self.mask_res == 128:
            self.up3 = Up_conv(in_channels = 320,
                                out_channels = 160,
                                kernel_size=2) # 64 -> 128 , channel 320 -> 160
        if self.mask_res == 256 :
            self.up4 = Up_conv(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2)  # 128 -> 256
        if self.mask_res == 512 :
            self.up4 = Up_conv(in_channels=160,
                               out_channels=160,
                               kernel_size=2)  # 128 -> 256
            self.up5 = Up_conv(in_channels = 160,
                                out_channels = 160,
                                kernel_size=2)
        self.outc = OutConv(160, n_classes)
    def forward(self, x16_out, x32_out, x64_out,
                x_init = None):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)    # 1,320,64,64
        if self.use_init_query :
            x_init = self.init_conv(x_init)   # 1, 320, 64, 64
            x = torch.cat([x, x_init], dim=1) # 1, 640, 64, 64
            x = self.double_conv(x)           # 1, 320, 64, 64
        x3_out = self.up3(x)  # 1,160,128,128
        x_in = x3_out
        if self.mask_res == 256 :
            x4_out = self.up4(x3_out)
            x_in = x4_out
        if self.mask_res == 512 :
            x4_out = self.up4(x3_out)
            x5_out = self.up5(x4_out)
            x_in = x5_out
        logits = self.outc(x_in)  # 1,4, 128,128
        return logits


class Segmentation_Head_d(nn.Module):

    def __init__(self,):
        super(Segmentation_Head_d, self).__init__()

        self.up = Up_conv(in_channels=3,
                          out_channels=3,
                          kernel_size=2)

    def forward(self, x):
        # x = [baatch, 64, 64, 3]
        x = self.up(x) # [batch, 128, 128, 77]
        # dim 3 to dim 1
        # x = x.mean(dim=3) # [batch, 128, 128]
        return x
