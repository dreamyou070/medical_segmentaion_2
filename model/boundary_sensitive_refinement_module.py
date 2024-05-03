import os
import torch
import numpy as np
import math
import cv2
import torchvision
from PIL import Image
from torch import nn
from typing import Dict, List, Optional, Set, Tuple, Union

class ViTSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        all_head_size = 320
        self.query = nn.Linear(320, all_head_size)
        self.key = nn.Linear(320, all_head_size)
        self.value = nn.Linear(320, all_head_size)
        self.dropout = nn.Dropout(0.0)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                cross_hidden_states,
                head_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        query_layer = self.query(hidden_states) # [1,64*64,160]
        key_layer = self.key(cross_hidden_states)
        value_layer = self.value(cross_hidden_states)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer) # [batch, len, dim]
        res = int(context_layer.shape[1] ** 0.5)
        context_layer = context_layer.reshape(1, res, res, -1).permute(0, 3, 1, 2).contiguous() # [1,160,64,64]
        return context_layer

class BoundarySensitive(nn.Module):
    def __init__(self, class_num) -> None:
        super().__init__()
        self.attn_16_module = ViTSelfAttention()
        self.attn_32_module = ViTSelfAttention()
        self.attn_64_module = ViTSelfAttention()
        self.attentions = [self.attn_16_module, self.attn_32_module, self.attn_64_module]
        self.outc = nn.Conv2d(320*3, class_num, kernel_size=1)

    def forward(self, edge_features, region_features, origin_features) :

        # [1]

        out_16_edge_ = self.attn_16_module(edge_features[0], region_features[0])  # [1,160,64,64]
        out_16_region_ = self.attn_16_module(region_features[0], edge_features[0])  # [1,160,64,64]
        out_16 = out_16_edge_ + out_16_region_ + origin_features[0]

        # [2]
        out_32_edge_ = self.attn_32_module(edge_features[1], region_features[1])
        out_32_region_ = self.attn_32_module(region_features[1], edge_features[1])
        out_32 = out_32_edge_ + out_32_region_ + origin_features[1]

        # [3]
        out_64_edge_ = self.attn_64_module(edge_features[2], region_features[2])
        out_64_region_ = self.attn_64_module(region_features[2], edge_features[2])
        out_64 = out_64_edge_ + out_64_region_ + origin_features[2]

        out = torch.cat([out_16, out_32, out_64], dim=1)
        logits = self.outc(out)  # 1,2,64,64
        return logits


"""
feature = torch.randn(1,2,256,256)
first_order_filter = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]]).float().unsqueeze(0).unsqueeze(0)
second_order_filter = torch.tensor([[-1, -1, -1],
                                   [-1, 8, -1],
                                   [-1, -1, -1]]).float().unsqueeze(0).unsqueeze(0)

# [1] apply gaussian blur : Laplacian over the Gaussian Filter
# [1.1] make Laplacian over Gaussian Filter (LoG)
# [1.2] apply LoG to the feature

def LoG(x,y,sigma) :
    norm = (x*x + y*y) / (2.0 * sigma**2)
    denominator = -1/(np.pi * sigma**4)
    exponential = math.exp(-norm)
    return ((1 - norm) * exponential) * denominator


# step 1. remove noise by applying gaussian blur

def gaussian_filter(sigma, size):
    # make gaussian filter with sigma and filter size 
    # [1] make 2D axis
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    # [2] gaussian function
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    torch_filder = torch.tensor(g).float().unsqueeze(0).unsqueeze(0)
    # [3] normalization
    return g / g.sum(), torch_filder

gausssian_filter, g_filter_torch = gaussian_filter(3, 3) # 1,1,3,3
g_filter_torch = g_filter_torch.expand(-1,2,-1,-1)          # 1,2,3,3
h_filter_torch = 1 - g_filter_torch
print(f'g_filter_torch : {g_filter_torch.shape}')
# feature = torch.randn(1,2,256,256)
# [1] sample

sample_dir = 'sample/6.png'
sample_pil = Image.open(sample_dir).resize((256,256))
sample_tensor = torchvision.transforms.ToTensor()(sample_pil).unsqueeze(0) # [1,1,256,256]
print(f'sample_tensor : {sample_tensor.shape}')
# [2] apply gaussian filter (
"""
"""
edge_feature = torch.randn(1,1,64,64)
region_feature = torch.randn(1,1,64,64)
x16_out, x32_out, x64_out = torch.randn(1,320,64,64), torch.randn(1,320,64,64), torch.randn(1,320,64,64)
x16_out_edge = (edge_feature * x16_out).reshape(1,320,-1).permute(0,2,1).contiguous() # [1,320,64,64] --> [1,64*64,320]
x16_out_region = (region_feature * x16_out).reshape(1,320,-1).permute(0,2,1).contiguous()
x32_out_edge = (edge_feature * x32_out).reshape(1,320,-1).permute(0,2,1).contiguous()
x32_out_region = (region_feature * x32_out).reshape(1,320,-1).permute(0,2,1).contiguous()
x64_out_edge = (edge_feature * x64_out).reshape(1,320,-1).permute(0,2,1).contiguous()
x64_out_region = (region_feature * x64_out).reshape(1,320,-1).permute(0,2,1).contiguous()


self_attn_module = ViTSelfAttention()

x16_out_edge_ = self_attn_module(x16_out_edge, x16_out_region) # [1,160,64,64]
x16_out_region_ = self_attn_module(x16_out_region, x16_out_edge) # [1,160,64,64]
x_16_out = x16_out_edge_ + x16_out_region_
print(f'x_16_out = {x_16_out.shape}')
x32_out_edge_ = self_attn_module(x32_out_edge, x32_out_region) # [1,160,64,64]
x32_out_region_ = self_attn_module(x32_out_region, x32_out_edge) # [1,160,64,64]
x_32_out = x32_out_edge_ + x32_out_region_
print(f'x_32_out = {x_32_out.shape}')
x64_out_edge_ = self_attn_module(x64_out_edge, x64_out_region) # [1,160,64,64]
x64_out_region_ = self_attn_module(x64_out_region, x64_out_edge) # [1,160,64,64]
x_64_out = x64_out_edge_ + x64_out_region_
print(f'x_64_out = {x_64_out.shape}')
total_out = torch.cat([x_16_out, x_32_out, x_64_out], dim=1)
print(f'total_out = {total_out.shape}')
"""
