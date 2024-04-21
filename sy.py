from torch import nn
import torch
import torch.nn.functional as F
class ReductionNet(nn.Module):
    def __init__(self, cross_dim, class_num):
        super(ReductionNet, self).__init__()
        self.layer = nn.Sequential(nn.Linear(cross_dim, class_num),
                                   nn.Softmax(dim=-1))
    def forward(self, x):
        class_embedding = x[:, 0, :]
        org_x = x[:, 1:, :]  # x = [1,196,768]
        reduct_x = self.layer(org_x)  # x = [1,196,3]
        reduct_x = reduct_x.permute(0, 2, 1).contiguous()  # x = [1,3,196]
        reduct_x = torch.matmul(reduct_x, org_x)  # x = [1,3,768]

        # normalizing in channel dimention ***
        reduct_x = F.normalize(reduct_x, p=2, dim=-1)
        if class_embedding.dim() != 3:
            class_embedding = class_embedding.unsqueeze(0)
        if class_embedding.dim() != 3:
            class_embedding = class_embedding.unsqueeze(0)
        x = torch.cat([class_embedding, reduct_x], dim=1)
        return x
"""
#reduction_net = ReductionNet(768, 3)
input = torch.randn(1, 4, 7)
input[:,0,:] = input[:,0,:] * 0
input[:,1,:] = input[:,1,:] * 0
linear_layer = nn.Linear(7, 3)
output = linear_layer(input)
print(output) # [1,4,3], 0,1 same
"""

weight_x = torch.randn((1,3,6))
weight_scale = torch.sum(weight_x, dim=-1)
print(f' (1) weight_x: {weight_x}')
print(f'weight_scale: {weight_scale}')
weight_x = weight_x / weight_scale.unsqueeze(-1)
print(f'after device, weight_x: {weight_x}')