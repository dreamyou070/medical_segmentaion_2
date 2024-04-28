import torch
import torch.nn as nn

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

    def __init__(self, channel1, channel2, n_classes):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.input_map = nn.Sequential(nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()
        self.n_classes = n_classes
        self.segment_head = nn.Conv2d(self.channel2,
                                      self.n_classes,
                                      kernel_size=1)


    def forward(self, x, y, in_map):
        # ------------------------------------------------------------------------------------------
        # [1] distraction discovering
        # if foreground and background is similar, distraction occurs
        # step 1.1 foreground and background division with nn.sigmoid
        foreground_map = self.input_map.to(x.device)(in_map)
        foreground_feature = x * foreground_map
        # 1.1 context exploration block
        fp = self.fp.to(x.device)(foreground_feature)

        # step 1.2 background feature
        background_map = 1 - foreground_map
        background_feature = x * background_map
        # 1.2 context exploration block
        fn = self.fn.to(x.device)(background_feature)

        # ------------------------------------------------------------------------------------------
        # [2] distrction removal
        # erase or remove what is bad
        refine1 = y - (self.alpha.to(x.device) * fp)
        refine1 = self.bn1.to(x.device)(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta.to(x.device) * fn)
        refine2 = self.bn2.to(x.device)(refine2)
        refine2 = self.relu2(refine2) # [1,320, 64, 64]
        output_map = self.output_map.to(x.device)(refine2)    # [1, 1, 64, 64]
        segment_out = self.segment_head.to(x.device)(refine2) # [1, 2, 64, 64]

        return segment_out, output_map

model = Focus(320, 320, 2)
x = torch.randn(1, 320, 128, 128)
y = torch.randn(1, 320, 128, 128)
in_map = torch.randn(1, 1, 64, 64)
segment_out, output_map = model(x, y, in_map)
print(segment_out.shape, output_map.shape)

