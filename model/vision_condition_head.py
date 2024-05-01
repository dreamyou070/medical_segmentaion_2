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


    def forward(self, x):

        x1 = x[0] # [batch, 64, 128, 128]
        x2 = x[1] # [batch, 128, 64, 64]
        x3 = x[2] # [batch, 320, 16, 16]
        x4 = x[3] # [batch, 512, 8, 8] # global feature


        x4 = x4.permute(0, 2, 3, 1) # batch, h, w, c
        x3 = x3.permute(0, 2, 3, 1) # batch, h, w, c
        x2 = x2.permute(0, 2, 3, 1) # batch, h, w, c
        x1 = x1.permute(0, 2, 3, 1) # batch, h, w, c

        x4 = x4.reshape(1, -1, 512)
        x3 = x3.reshape(1, -1, 320)
        x2 = x2.reshape(1, -1, 128)
        x1 = x1.reshape(1, -1, 64)

        y4 = self.fc_4(x4)  # [batch pixnum, 512] -> [batch, 8*8, 768]
        y3 = self.fc_3(x3)  # [batch pixnum, 320] -> [batch, 16*16, 768]
        y2 = self.fc_2(x2)  # [batch pixnum, 128] -> [batch, 32*32, 768]
        y1 = self.fc_1(x1)  # [batch pixnum, 64]  -> [batch, 64*64, 768]

        condition_dict = {}

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