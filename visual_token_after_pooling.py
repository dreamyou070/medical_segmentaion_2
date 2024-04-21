import os
import torch

visual_embedding = torch.randn(1,197,768)
cls_token = visual_embedding[:,0,:]
context_token = visual_embedding[:,1:,:]
pooling_layer = torch.nn.AdaptiveAvgPool2d((1,1))
pooled_output = pooling_layer(context_token.permute(0,2,1)).squeeze(2)
print(pooled_output.shape)
#pooled_output = torch.cat([cls_token, pooled_output], dim=1)

