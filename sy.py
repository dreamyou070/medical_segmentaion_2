from torch import nn
import torch
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
init_query = torch.randn(1,160,256,256)
in_dim = 160
query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
proj_query = query_conv(init_query).view(1, -1, 256 * 256).permute(0, 2, 1) # [1, 256, 256, 160] -> [1, 256*256, 160]
proj_key = key_conv(init_query).view(1, -1, 256 * 256) # [1, 256, 256, 160] -> [1, 256*256, 160]
energy = torch.bmm(proj_query, proj_key)
attention = nn.Softmax(dim=-1)(energy) # [batch, pix_num, pix
print(attention.shape) # torch.Size([1, 65536, 65536])

value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
proj_value = value_conv(init_query).view(1, -1, 256 * 256) # [B,C,(HxW)] -> [B,(HxW),C]
print(proj_value.shape) # torch.Size([1, 160, 65536])
out = torch.bmm(proj_value, attention.permute(0, 2, 1))