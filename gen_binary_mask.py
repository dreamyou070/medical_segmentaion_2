import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch

gt_path = 'sample_200.npy'
gt_arr = np.load(gt_path)  # 256,256 (brain tumor case)
binary_gt = np.where(gt_arr > 0, 255, 0)
binary_gt = Image.fromarray(binary_gt.astype(np.uint8))
binary_gt = binary_gt.resize((224,224), Image.BICUBIC) # only object
binary_gt = np.array(binary_gt)
binary_gt = np.where(binary_gt > 100, 1, 0) # binary mask
# mask_embedding
kernel_size = 16
stride =16
mask_embedding = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
mask_embedding.weight.data.fill_(1)
mask_embedding.weight.requires_grad = False
mask_embedding = mask_embedding(torch.tensor(binary_gt).unsqueeze(0).float()) # [1,14,14]
# 3dim to 2 dim
cls_embedding = torch.ones((1,1))
mask_embedding = mask_embedding.view(1, -1).contiguous() # 1, 196
mask_embedding = torch.cat([cls_embedding, mask_embedding], dim=1).unsqueeze(dim=-1) # 1, 197,1
mask_embedding = mask_embedding.squeeze(0) # 1, 197, 1
# 1,197 * 1,197*768

# 224 -> 14
print(mask_embedding.shape)