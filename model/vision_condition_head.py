from torch import nn
class vision_condition_head(nn.Module):

    def __init__(self, reverse=False, use_one=False):
        super(vision_condition_head, self).__init__()
        multi_dims = [64, 128, 320, 512]
        condition_dim = 768

        self.fc_1 = nn.Linear(multi_dims[0], condition_dim)
        self.fc_2 = nn.Linear(multi_dims[1], condition_dim)
        self.fc_3 = nn.Linear(multi_dims[2], condition_dim)
        self.fc_4 = nn.Linear(multi_dims[3], condition_dim)

        self.reverse = reverse
        self.use_one = use_one

    def forward(self, x):

        x1 = x[0].permute(0, 2, 3, 1)
        x2 = x[1].permute(0, 2, 3, 1)
        x3 = x[2].permute(0, 2, 3, 1)
        x4 = x[3].permute(0, 2, 3, 1)

        x1 = x1.reshape(1, -1, 64)
        x2 = x2.reshape(1, -1, 128)
        x3 = x3.reshape(1, -1, 320)
        x4 = x4.reshape(1, -1, 512)

        y1 = self.fc_1(x1)  # batch, pixnum, 768
        y2 = self.fc_2(x2)  # batch, pixnum, 768
        y3 = self.fc_3(x3)  # batch, pixnum, 768
        y4 = self.fc_4(x4)  # batch, pixnum, 768 (much deep features)

        condition_dict = {}

        if self.reverse :
            condition_dict[64] = y1
            condition_dict[32] = y2
            condition_dict[16] = y3
            condition_dict[8] = y4

        if not self.reverse :

            elif self.use_one :
                condition_dict[64] = y3
                condition_dict[32] = y3
                condition_dict[16] = y3
            else :
                condition_dict[64] = y4
                condition_dict[32] = y3
                condition_dict[16] = y2
                condition_dict[8] = y1

        return condition_dict