import os
import IPython.display
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
from utils import utils_ade20k

DATASET_PATH = 'dataset/'
index_file = 'index_ade20k.pkl'
with open(f'{DATASET_PATH}/{index_file}', 'rb') as f:
    index_ade20k = pkl.load(f)

print("File loaded, description of the attributes:")
print('--------------------------------------------')
for attribute_name, desc in index_ade20k['description'].items():
    print('* {}: {}'.format(attribute_name, desc))
print('--------------------------------------------\n')