import torch
from sklearn.metrics import auc, roc_curve


def dice_coeff(real_mask: torch.Tensor, smooth=0.000001, mse=None):
    # real_mask = binary mask map
    # mse = anomaly map
    intersection = torch.sum(mse * real_mask, dim=[1, 2]) # only different value in anomal pixel region
    union = torch.sum(mse, dim=[1, 2]) + torch.sum(real_mask, dim=[1, 2]) # dice = intersection / union
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
    return dice

def IoU(real, recon):
    import numpy as np
    real = real.cpu().numpy()
    recon = recon.cpu().numpy()
    intersection = np.logical_and(real, recon)
    union = np.logical_or(real, recon)
    return np.sum(intersection) / (np.sum(union) + 1e-8)

def precision(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FP = ((real_mask == 1) & (recon_mask == 0))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)



def recall(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FN = ((real_mask == 0) & (recon_mask == 1))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)


def FPR(real_mask, recon_mask):
    FP = ((real_mask == 1) & (recon_mask == 0))
    TN = ((real_mask == 0) & (recon_mask == 0))
    return torch.sum(FP).float() / ((torch.sum(FP) + torch.sum(TN)).float() + 1e-6)


def ROC_AUC(real_mask, square_error):
    if type(real_mask) == torch.Tensor:
        return roc_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())

def AUC_score(fpr, tpr):
    return auc(fpr, tpr)