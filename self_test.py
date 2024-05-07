import os, torch

self_attn_map = torch.randn((1, 64*64, 64*64))
#self_attn_map = self_attn_map.view(-1, 64, 64, 64, 64)
#print(self_attn_map.size())
sigmoid_layer = torch.nn.Sigmoid()
self_attn_map = sigmoid_layer(self_attn_map)
f_feature = (self_attn_map)
b_feature = (1 - self_attn_map)