import numpy as np
from medpy import metric

def score_multi(target_class_idx, confusion_score_np):
    total_tp = 0
    total_true, total_pred = 0, 0
    for idx in target_class_idx:
        tp = confusion_score_np[idx, idx]
        total_tp += tp
        true_value = confusion_score_np[idx, :].sum()
        pred_value = confusion_score_np[:, idx].sum()
        total_true += true_value
        total_pred += pred_value
    core = 2 * total_tp / (total_true + total_pred)
    return core

def evaluate_braTS_dict (confusion_score, target_class_idx, label_list) :

    confusion_score_np = np.array(confusion_score)
    score_dict = {}
    for trg_class_list, label in zip(target_class_idx, label_list):
        score = score_multi(trg_class_list, confusion_score_np)
        score_dict[label] = score
    return score_dict

# ----------------------------------------------------------------------------------------------
def hd95_score(masks_pred, gt_arr, target_class_ids, label_list, hd95_dict):
    bool_pred = np.zeros_like(masks_pred)
    bool_gt = np.zeros_like(gt_arr)
    for target_class_list, label in zip(target_class_ids, label_list):
        for i in target_class_list:
            bool_pred[masks_pred == i] = 1
            bool_gt[gt_arr == i] = 1
        hd95 = metric.binary.hd95(bool_pred, bool_gt)
        if label not in hd95_dict.keys():
            hd95_dict[label] = []
        hd95_dict[label].append(hd95)
    return hd95_dict