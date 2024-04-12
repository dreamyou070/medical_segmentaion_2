import os
import torch
attn_map = torch.randn(1,64*64, 77)
attn_map = attn_map[:,:, :4]
print(attn_map.shape)