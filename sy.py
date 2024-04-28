from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from tensorflow.keras.utils import to_categorical

gt_64_array = np.zeros((64,64))
gt_64_array = np.where(gt_64_array > 100, 1, 0) # [64,64]
gt_64_array = to_categorical(gt_64_array, num_classes=2) # [64,64,2]
print(type(gt_64_array)) # [64,64,2]

# np -> torch
gt_64_tensor = torch.tensor(gt_64_array) # [64,64,2]