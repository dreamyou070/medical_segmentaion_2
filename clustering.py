import os
import torch

class_0_embedding = torch.nn.Parameter(torch.randn(768))
class_1_embedding = torch.nn.Parameter(torch.randn(768))
class_2_embedding = torch.nn.Parameter(torch.randn(768))

# [1]
raw_img = torch.randn(1,4,64,64)
raw_mask = torch.randn(1,3,256,256)
