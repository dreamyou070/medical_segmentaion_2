from torch import nn
import torch


trainable_params = [{'params': 'a',
                         'lr': 'b'}]
trainable_params += [{'params': 'c',
                          'lr': 'd'}]
print(len(trainable_params))