import torch


# out = [batch, C, H, W]
gt = torch.randn((1,4,128,128))
out = torch.randn((1,4,128,128))

class_num = gt.shape[1]
model_dim = out.shape[1]
class_wise_mean = []
for i in range(class_num):
    gt_map = gt[:, i, :, :].repeat(1,model_dim, 1, 1) # 0 = non, 1 = class pixel
    classwise_map = gt_map * out
    pixel_num = gt_map.sum()
    if pixel_num != 0:
        class_mean = classwise_map.sum() / pixel_num
        class_wise_mean.append(class_mean)