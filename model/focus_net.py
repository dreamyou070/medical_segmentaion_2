""" complex decoder """
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.segmentation_unet import Up_conv, OutConv


class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()

        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        # [1] channel reduction
        self.p1_channel_reduction = nn.Sequential(nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
                                                  nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
                                                  nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
                                                  nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
                                                  nn.BatchNorm2d(self.channels_single), nn.ReLU())
        # [2] convolution and deconvolution
        self.p1 = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
                                nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
                                nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
                                nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4 = nn.Sequential(nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
                                nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        # [3] fusion
        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                       nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        # [1] upscaling
        up = self.up(y)

        # [2] in_map controlling
        input_map = self.input_map(in_map) # upscaling (from low resolution to high resolution)
        # [3.1] foreground focus attn
        f_feature = x * input_map
        fp = self.fp(f_feature)
        # [3.2] background focus attn
        b_feature = x * (1 - input_map)
        fn = self.fn(b_feature)

        # [4] refine
        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map


class PFNet(nn.Module):
    def __init__(self, mask_res, n_classes, use_layer_norm=False,):
        super(PFNet, self).__init__()

        # focus
        self.focus1 = Focus(640, 1280)
        self.focus2 = Focus(320, 640)

        self.use_layer_norm = use_layer_norm

        # Segmentation Head
        if self.use_layer_norm:
            if mask_res == 128:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160 , 128, 128]),)
            if mask_res == 256:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160*3, 128, 128]),
                                                       Up_conv(in_channels=160, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160, 256,256]))
            if mask_res == 512:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160*3, 128, 128]),
                                                       Up_conv(in_channels=160, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160*2, 256, 256]),
                                                       Up_conv(in_channels=160, out_channels=160, kernel_size=2),
                                                       nn.LayerNorm([160, 512, 512]))
        else :
            if mask_res == 128:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320, out_channels=160, kernel_size=2),
                                                       )
            if mask_res == 256:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320, out_channels=160, kernel_size=2),
                                                       Up_conv(in_channels=160, out_channels=160, kernel_size=2),)
            if mask_res == 512:
                self.segmentation_head = nn.Sequential(Up_conv(in_channels=320, out_channels=160, kernel_size=2),
                                                       Up_conv(in_channels=160, out_channels=160, kernel_size=2),
                                                       Up_conv(in_channels=160, out_channels=160, kernel_size=2),)
        self.outc = OutConv(160, n_classes)



    def gen_feature(self, x16_out, x32_out, x64_out, high_semantic_map):
        # x16_out = [batch, 1280, 16, 16]
        # x32_out = [batch, 640, 32, 32]
        # x64_out = [batch, 320, 64, 64]
        # high_semantic_map = [batch, 1, 16, 16]
        focus3, predict3 = self.focus1(x=x32_out,
                                       y=x16_out,
                                       in_map=high_semantic_map)
        focus2, predict2 = self.focus2(x=x64_out, y=focus3, in_map=predict3)
        focus = focus2 * predict2  # focus = [1, 320, 64, 64]
        x = self.segmentation_head(focus)
        return x

    def segment_feature(self, x):

        x = self.outc(x)
        return x

