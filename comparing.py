import torch


pred_label = torch.tensor([0.1,0.2])
value = torch.softmax(pred_label, dim=0)
mask_pred_argmax = torch.argmax(value).item()
print(mask_pred_argmax)