from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
import torch.nn.functional as F
import torch
def eval_step(engine, batch):
    return batch

def generate_confusion_matrix (y_pred, y_true, class_num) :


    evaluator = Engine(eval_step)

    metric = ConfusionMatrix(num_classes=class_num)
    # [1] confusion, dice index
    metric.attach(evaluator, 'cm')
    metric.attach(evaluator, 'dice')

    # [2] calculate
    state = evaluator.run([[y_pred, y_true]])

    # [3] get value
    confusion_matrix = state.metrics['cm']
    dice_coeff = state.metrics['dice']
    return confusion_matrix, dice_coeff

def get_IoU(y_pred, y_true, class_num, confusion_matrix):

    if confusion_matrix is None :
        evaluator = Engine(eval_step)
        confusion_matrix, dice_coeff = generate_confusion_matrix (evaluator, y_pred, y_true, class_num)

    real_axis, pred_axis = confusion_matrix.shape
    IoU_dict, IoU_list = {}, []

    for real_idx in range(real_axis) :
        TP = confusion_matrix[real_idx, real_idx]
        FP = confusion_matrix[real_idx, :].sum() - TP
        FN = confusion_matrix[:, real_idx].sum() - TP
        IoU = TP / (TP + FP + FN)
        if type(IoU) == torch.Tensor:
            IoU_dict[real_idx] = IoU.item()
        else:
            IoU_dict[real_idx] = IoU
        IoU_list.append(IoU)
    mIOU = sum(IoU_list) / len(IoU_list)
    return IoU_dict, mIOU

def tversky_loss(y_pred, y_true, class_num, confusion_matrix,
                 alpha = 0.3, beta = 0.7,):
    # alpha = coeff for FN
    # beta = coeff for FP

    if confusion_matrix is None :
        evaluator = Engine(eval_step)
        confusion_matrix, dice_coeff = generate_confusion_matrix (evaluator, y_pred, y_true, class_num)

    real_axis, pred_axis = confusion_matrix.shape
    tversky_dict = {}
    for real_idx in range(real_axis):
        TP = confusion_matrix[real_idx, real_idx]
        FP = confusion_matrix[real_idx, :].sum() - TP
        FN = confusion_matrix[:, real_idx].sum() - TP
        IoU = TP / (TP + beta * FP + alpha * FN)
        tversky_dict[real_idx] = IoU
    return tversky_dict