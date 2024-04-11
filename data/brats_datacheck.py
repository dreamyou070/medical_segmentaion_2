import os
import numpy as np

trg_folder = '/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation/test/res_256/mask_128'
files = os.listdir(trg_folder)
for file in files :
    file_path = os.path.join(trg_folder, file)
    mask_arr = np.load(file_path)
    print(f'mask shape (128,128) : {mask_arr.shape}')