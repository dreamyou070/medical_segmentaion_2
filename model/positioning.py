import torch
import torch.nn as nn


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

    def __init__(self, use_channel_attn, use_self_attn=False):
        super(AllPositioning, self).__init__()

        self.position_net = {}
        for layer_name in self.layer_names.keys():

            print(f'self.self_layer_names[layer_name] = {self.self_layer_names[layer_name]}')

            #if use_self_attn:
            #    self.position_net[layer_name] = Positioning(channel = int(self.self_layer_names[layer_name]),
            #                                                use_channel_attn = use_channel_attn)
            #else:
            #    self.position_net[layer_name] = Positioning(channel = int(self.layer_names[layer_name]),
            #                                                use_channel_attn=use_channel_attn)

    def forward(self, x, layer_name):
        net = self.position_net[layer_name]
        # generate spatial attention result and global_feature
        x, global_feat = net(x)
        return x, global_feat


#x16_out = torch.randn(1, 1280, 16, 16)
#x32_out = torch.randn(1, 640, 32, 32)
#x64_out = torch.randn(1, 320, 64, 64)

#y16_out = net(x16_out, 'up_blocks_1_attentions_2_transformer_blocks_0_attn2')
#y32_out = net(x32_out, 'up_blocks_2_attentions_2_transformer_blocks_0_attn2')
#y64_out = net(x64_out, 'up_blocks_3_attentions_2_transformer_blocks_0_attn2')