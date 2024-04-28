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

    def __init__(self, channel1, channel2, n_classes, do_up):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.do_up = do_up
        if self.do_up:
            self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                           nn.Sigmoid())
        else :
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




###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1) # dimension shrinking
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)   #
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)      # dim reconstruction
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # [2]
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height) # [B,C,(HxW)] -> [B,(HxW),C]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel, use_channel_attn):
        super(Positioning, self).__init__()
        self.channel = channel
        self.use_channel_attn = use_channel_attn
        if self.use_channel_attn:
            self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map =  nn.Conv2d(self.channel, 1, 11, 1, 5)  # as if average pooling

    def forward(self, x):
        if self.use_channel_attn:
            self.cab = self.cab.to(x.device)
            x = self.cab(x) # is like self attntion
        self.sab = self.sab.to(x.device)
        sab = self.sab(x)
        self.map = self.map.to(x.device)
        global_sab = self.map(sab)
        return sab+global_sab, global_sab

class AllPositioning(nn.Module):


    layer_names = {'up_blocks_1_attentions_2_transformer_blocks_0_attn2':1280,
                   'up_blocks_2_attentions_2_transformer_blocks_0_attn2': 640,
                   'up_blocks_3_attentions_2_transformer_blocks_0_attn2': 320,}

    self_layer_names = {'up_blocks_1_attentions_2_transformer_blocks_0_attn1' : 1280,
                        'up_blocks_2_attentions_2_transformer_blocks_0_attn1' : 640,
                        'up_blocks_3_attentions_2_transformer_blocks_0_attn1' : 320,}

    def __init__(self, use_channel_attn, use_self_attn=False, n_classes=2):
        super(AllPositioning, self).__init__()

        # ------------------------------------------------------------------------------------------
        # [1] position network
        self.position_net = {}
        if use_self_attn:
            layer_names = self.self_layer_names
        else:
            layer_names = self.layer_names
        for layer_name in layer_names.keys():
            self.position_net[layer_name] = Positioning(channel = int(layer_names[layer_name]),
                                                        use_channel_attn = use_channel_attn)
        # ------------------------------------------------------------------------------------------
        # [2] focus network
        self.focus_net = {}
        for layer_name in layer_names.keys():
            if 'up_blocks_1' in layer_name:
                do_up = False
            else :
                do_up = True
            self.focus_net[layer_name] = Focus(channel1 = int(layer_names[layer_name]),
                                               channel2 = int(layer_names[layer_name]),
                                               n_classes = n_classes,
                                               do_up = do_up)

    def forward(self, x, layer_name):
        net = self.position_net[layer_name]
        # generate spatial attention result and global_feature
        x, global_feat = net(x)
        return x, global_feat

    def predict_seg(self,channel_attn_query, spatial_attn_query, layer_name, in_map) :
        focus_net = self.focus_net[layer_name]
        mask_pred, focus_map = focus_net(channel_attn_query, spatial_attn_query, in_map)
        return mask_pred, focus_map




#x16_out = torch.randn(1, 1280, 16, 16)
#x32_out = torch.randn(1, 640, 32, 32)
#x64_out = torch.randn(1, 320, 64, 64)

#y16_out = net(x16_out, 'up_blocks_1_attentions_2_transformer_blocks_0_attn2')
#y32_out = net(x32_out, 'up_blocks_2_attentions_2_transformer_blocks_0_attn2')
#y64_out = net(x64_out, 'up_blocks_3_attentions_2_transformer_blocks_0_attn2')



"""
x = torch.randn(1,320,64,64)
y = torch.randn(1,320,64,64)
in_map = torch.randn(1,1,64,64)
model = Focus(320, 320)
refine2, output_map = model(x=x, y=y, in_map=None)
print(refine2.shape)
print(output_map.shape)
"""