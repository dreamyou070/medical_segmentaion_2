import numpy as np
import torch
import os
from PIL import Image
"""
def torch_to_pil(torch_img, binary=False):
    # torch_img = [3, H, W], from -1 to 1
    if torch_img.dim() == 3:
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
    else:
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8)

    if binary :
        np_img = np.where(np_img > 127, 255, 0).astype(np.uint8)
        pil = Image.fromarray(np_img).convert("L").convert('RGB')
    else :
        pil = Image.fromarray(np_img).convert("RGB")
    return pil

def gaussian_filter(sigma, size):
    # make gaussian filter with sigma and filter size 
    # [1] make 2D axis
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    # [2] gaussian function
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    torch_filder = torch.tensor(g).float().unsqueeze(0).unsqueeze(0)
    # [3] normalization
    return g / g.sum(), torch_filder

gausssian_filter, g_filter_torch = gaussian_filter(3, 3)
g_filter_torch = g_filter_torch.expand(-1, 2, -1, -1)  # [1,2,3,3]
h_filter_torch = 1 - g_filter_torch                    # [1,2,3,3]

# -------------------------------------
output = torch.randn(1, 2, 64,64)
edge_feature = torch.nn.functional.conv2d(output, g_filter_torch, padding=1)
region_feature = torch.nn.functional.conv2d(output, h_filter_torch, padding=1)
background_feature = output - region_feature - edge_feature
"""
def structure_loss(pred, mask # gt
                   ):

    from torch.nn import functional as F
    # [0] weight
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # [1] bce loss
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    # [2] pred and mask
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean() # one value

pred = torch.randn(1,2,64,64)
mask = torch.randn(1,2,64,64)
loss = structure_loss(pred, mask)
print(loss)

