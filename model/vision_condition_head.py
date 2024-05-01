from torch import nn
import torch
class vision_condition_head(nn.Module):

    def __init__(self, reverse=False, use_one=False):
        super(vision_condition_head, self).__init__()
        """ make 768 cross dim condition """
        multi_dims = [64, 128, 320, 512]
        condition_dim = 768
        self.fc_1 = nn.Linear(multi_dims[0], condition_dim)
        self.fc_2 = nn.Linear(multi_dims[1], condition_dim)
        self.fc_3 = nn.Linear(multi_dims[2], condition_dim)
        self.fc_4 = nn.Linear(multi_dims[3], condition_dim) # deep feature

        self.reverse = reverse
        self.use_one = use_one
        self.feature_up_4_3 = nn.Sequential(nn.Conv2d(512, 320, 7, 1, 3),
                                       nn.BatchNorm2d(320),
                                       nn.UpsamplingBilinear2d(scale_factor=2))
        self.feature_up_3_2 = nn.Sequential(nn.Conv2d(320, 128, 7, 1, 3),
                                       nn.BatchNorm2d(128),
                                       nn.UpsamplingBilinear2d(scale_factor=2))
        self.feature_up_2_1 = nn.Sequential(nn.Conv2d(128, 64, 7, 1, 3),
                                       nn.BatchNorm2d(64),
                                       nn.UpsamplingBilinear2d(scale_factor=2))
        # --------------------------------------------------------------------------------------------------------
        self.feature_4_conv = nn.Sequential(nn.Conv2d(640, 320, kernel_size=3, stride=1, padding=1, dilation=1),
                                       nn.Sigmoid())
        self.feature_3_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1),
                                       nn.Sigmoid())
        self.feature_2_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                                       nn.Sigmoid())


    def forward(self, x):

        x1 = x[0] # [batch, 64, 128, 128]
        x2 = x[1] # [batch, 128, 64, 64]
        x3 = x[2] # [batch, 320, 16, 16]
        x4 = x[3] # [batch, 512, 8, 8] # global feature

        x43 = self.feature_up_4_3(x4)        # [batch, 320, 16, 16]
        x4_out = torch.cat([x43, x3], dim=1) #
        x4_out = self.feature_4_conv(x4_out) # [batch, 320, 16, 16]

        x32 = self.feature_up_3_2(x3) # [batch, 128, 32, 32]
        x3_out = torch.cat([x32, x2], dim=1)
        x3_out = self.feature_3_conv(x3_out) # [batch, 128, 32, 32]

        x21 = self.feature_up_3_2(x2) # [batch, 64, 64, 64]
        x2_out = torch.cat([x21, x1], dim=1)
        x2_out = self.feature_2_conv(x2_out) # [batch, 64, 64, 64]

        #x1 = x[0].permute(0, 2, 3, 1) # batch, h, w, c
        #x2 = x[1].permute(0, 2, 3, 1)
        #x3 = x[2].permute(0, 2, 3, 1)
        x4 = x4.permute(0, 2, 3, 1) # batch, h, w, c
        x3 = x4_out.permute(0, 2, 3, 1) # batch, h, w, c
        x2 = x3_out.permute(0, 2, 3, 1) # batch, h, w, c
        x1 = x2_out.permute(0, 2, 3, 1) # batch, h, w, c

        x4 = x4.reshape(1, -1, 512)
        x3 = x3.reshape(1, -1, 320)
        x2 = x2.reshape(1, -1, 128)
        x1 = x1.reshape(1, -1, 64)

        y4 = self.fc_4(x4)  # [batch pixnum, 512] -> [batch, 8*8, 768]
        y3 = self.fc_3(x3)  # [batch pixnum, 320] -> [batch, 16*16, 768]
        y2 = self.fc_2(x2)  # [batch pixnum, 128] -> [batch, 32*32, 768]
        y1 = self.fc_1(x1)  # [batch pixnum, 64]  -> [batch, 64*64, 768]

        #y4 = self.fc_4(x4)  # [batch pixnum, 512] -> [batch, 8*8, 768] --> channel expanding ...

        # upscaling

        condition_dict = {}


        # why don't i sigmoid to extract only region specific feature ?


        if self.reverse :

            condition_dict[64] = y1
            condition_dict[32] = y2
            condition_dict[16] = y3 # really deep feature here !
            condition_dict[8]  = y4 # really deep feature here !

        else :
            #condition_dict[64] = y4
            #condition_dict[32] = y3
            #condition_dict[16] = y2
            #condition_dict[8] = y1
            condition_dict[64] = y3
            condition_dict[32] = y2
            condition_dict[16] = y1
            condition_dict[8] = y1

        return condition_dict