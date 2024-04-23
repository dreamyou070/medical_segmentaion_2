import torch
import torch.nn as nn
import torch.nn.functional as F
class ReductionNet(nn.Module):
    def __init__(self, cross_dim, class_num,
                 use_weighted_reduct):

        super(ReductionNet, self).__init__()
        self.layer = nn.Sequential(nn.Linear(cross_dim, class_num),
                                   nn.Softmax(dim=-1))
        self.use_weighted_reduct = use_weighted_reduct

    def forward(self, x):
        class_embedding = x[:, 0, :]
        org_x = x[:, 1:, :]  # x = [1,196,768]

        reduct_x = self.layer(org_x)  # x = [1,196,3]
        if self.use_weighted_reduct:
            weight_x = reduct_x.permute(0, 2, 1).contiguous()  # x = [1,3,196]
            weight_scale = torch.sum(weight_x, dim=-1)
            reduct_x = torch.matmul(weight_x, org_x)  # x = [1,3,768]
            # normalizing in channel dimention ***
            # reduct_x = F.normalize(reduct_x, p=2, dim=-1)
            reduct_x = reduct_x / weight_scale.unsqueeze(-1)
        else:
            reduct_x = reduct_x.permute(0, 2, 1).contiguous()  # x = [1,3,196]
            reduct_x = torch.matmul(reduct_x, org_x)  # [1,3,196] [1,196,768] = [1,3,768]
            reduct_x = F.normalize(reduct_x, p=2, dim=-1)

        if class_embedding.dim() != 3:
            class_embedding = class_embedding.unsqueeze(0)
        if class_embedding.dim() != 3:
            class_embedding = class_embedding.unsqueeze(0)

        x = torch.cat([class_embedding, reduct_x], dim=1)

        return x