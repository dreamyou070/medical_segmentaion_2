import torch, os

trg_attention = torch.randn(64*64,2)
max_value = torch.max(trg_attention, dim=0).values
print(max_value)