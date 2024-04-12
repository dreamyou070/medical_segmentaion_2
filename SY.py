import torch
from torch import nn
attn_map = torch.randn(1, 16*16, 4)
original_res = int(attn_map.shape[1] ** 0.5)                     # trg_res = 64
target_res = 64
upscale_factor = target_res // original_res
# [b,pix_num, dim -> b, res, res, dim]
original_map = attn_map.view(attn_map.shape[0], original_res, original_res, attn_map.shape[2]).permute(0, 3, 1, 2).contiguous()
# upscaling
print(f'original map shape: {original_map.shape}')
attn_map = nn.functional.interpolate(original_map,
                                     scale_factor=upscale_factor,
                                     mode='bilinear',
                                     align_corners=False)
print(attn_map.shape)