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
                 mask_res = 128):

        super(SemanticSeg_Gen, self).__init__()

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
        self.outc = OutConv(160, n_classes)


    def reconstruction(self, x):
        return self.decoder_model(x)

    def forward(self, x16_out, x32_out, x64_out):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)       # 1,320,64,64

        # ----------------------------------------------------------------------------------------------------
        # semantic rich feature
        x = self.segmentation_head(x)
        logits = self.outc(x)  # 1, 4, 128,128

        return logits #, gen_feature


class SemanticSeg_Gen(nn.Module):
    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_batchnorm=True,
                 use_instance_norm = True,
                 mask_res = 128,
                 high_latent_feature = False,
                 init_latent_p = 1):

        super(SemanticSeg_Gen, self).__init__()

        c = 320

        self.mlp_layer_1 = torch.nn.Linear(1280, c)
        self.upsample_layer_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.mlp_layer_2 = torch.nn.Linear(640, c)
        self.upsample_layer_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.mlp_layer_3 = torch.nn.Linear(320, c)
        self.upsample_layer_3 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)

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

        x16_out = self.dim_and_res_up(self.mlp_layer_1, self.upsample_layer_1, x16_out)
        x32_out = self.dim_and_res_up(self.mlp_layer_2, self.upsample_layer_2, x32_out)
        x64_out = self.dim_and_res_up(self.mlp_layer_3, self.upsample_layer_3, x64_out)
        x = torch.cat([x16_out, x32_out, x64_out], dim=1)
        x = self.segmentation_head(x)
        logits = self.outc(x)  # 1, 4, 128,128

        return logits


class SemanticSeg_2(nn.Module):
    def __init__(self,
                 n_classes,
                 bilinear=False,
                 use_batchnorm=True,
                 use_instance_norm=True,
                 mask_res=128,
                 high_latent_feature=False,
                 init_latent_p=1):
        super(SemanticSeg_2, self).__init__()

        c = 320

        self.mlp_layer_1 = torch.nn.Linear(1280, c)
        self.upsample_layer_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.mlp_layer_2 = torch.nn.Linear(640, c)
        self.upsample_layer_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.mlp_layer_3 = torch.nn.Linear(320, c)
        self.upsample_layer_3 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)

        # self.outc = OutConv(160, n_classes)
        if mask_res == 128:
            self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160, kernel_size=2))
        if mask_res == 256:
            self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160*3, kernel_size=2),
                                                   Up_conv(in_channels=160*3, out_channels=160, kernel_size=2))
        if mask_res == 512:
            self.segmentation_head = nn.Sequential(Up_conv(in_channels=320*3, out_channels=160*3, kernel_size=2),
                                                   Up_conv(in_channels=160*3, out_channels=160*2, kernel_size=2),
                                                   Up_conv(in_channels=160*2, out_channels=160, kernel_size=2))
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

        x = self.segmentation_head(x)
        logits = self.outc(x)  # 1, 4, 128,128
        return logits

        # x = self.segmentation_head(x)
        # logits = self.outc(x)  # 1, 4, 128,128

        # return logits  # , gen_feature




"""
# [1] get image
image_path = 'data_sample/image/sample_200.jpg'
image = Image.open(image_path).convert('RGB')

# [2] make pipeline
image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
inputs = image_processor(images=image, return_tensors="pt") # shape of pixel_values: torch.Size([1, 3, 512,512])


# [3] main model
# mit-b0 is base model and not for segmentation model
# model = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
# SegformerForSemanticSegmentation
semantic_segmentation_0 = pipeline(task = "image-segmentation", model = "nvidia/mit-b0").model
semantic_segmentation_1 = pipeline(task = "image-segmentation", model = "nvidia/mit-b1").model
semantic_segmentation_2 = pipeline(task = "image-segmentation", model = "nvidia/mit-b2").model
semantic_segmentation_3 = pipeline(task = "image-segmentation", model = "nvidia/mit-b3").model
semantic_segmentation_4 = pipeline(task = "image-segmentation", model = "nvidia/mit-b4").model
semantic_segmentation_5 = pipeline(task = "image-segmentation", model = "nvidia/mit-b5").model

out_0 = semantic_segmentation_0(pixel_values = inputs['pixel_values'])
out_1 = semantic_segmentation_1(pixel_values = inputs['pixel_values'])
out_2 = semantic_segmentation_2(pixel_values = inputs['pixel_values'])
out_3 = semantic_segmentation_3(pixel_values = inputs['pixel_values'])
out_4 = semantic_segmentation_4(pixel_values = inputs['pixel_values'])
out_5 = semantic_segmentation_5(pixel_values = inputs['pixel_values'])
"""
"""

output = model(pixel_values = inputs['pixel_values'])

print(output.logits.shape)

# [3] segformer model
#input_torch = torch.randn(1, 3, 1024, 1024)
# SegformerForSemanticSegmentation

#
#from datasets import load_dataset

#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]


model = SegformerModel.from_pretrained("nvidia/mit-b0")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
[1, 256, 16, 16]
"""