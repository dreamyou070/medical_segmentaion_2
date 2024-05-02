import numpy as np

eps = 1e-15

def generate_Iou(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection + eps
    IoU = intersection / union
    return IoU

def generate_Dice(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set + eps
    Dice = (2 * intersection) / union
    return Dice
