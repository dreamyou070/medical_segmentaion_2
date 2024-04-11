import os
import h5py
import numpy as np
import nibabel as nib
import glob
from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# [1] scaler to change image pixel range
# [2]
# [3] slicing on slice axis
# [4] combine and reshape to remove bad point (240 -> 128)

def scaling_img(raw_img, scaler):
    raw = raw_img.reshape(-1, raw_img.shape[-1])  # [all pixel values, angles]
    raw = scaler.fit_transform(raw)  # set value from 0 ~ 1
    raw_img = raw.reshape(raw_img.shape)  # 240,240,155
    return raw_img

def main():

    # [1] flair
    # [2] t1ce
    # [3] t2
    # [4] mask
    """
    folder_128 = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_128/train/anomal'
    folder_128_mask = os.path.join(folder_128, 'mask_128')
    folder_256_mask = os.path.join(folder_128, 'mask_256')
    os.makedirs(folder_256_mask, exist_ok = True)
    """
    test_folder_128 = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_128/test/anomal'
    test_folder_128_mask = os.path.join(test_folder_128, 'mask_128')
    test_folder_256_mask = os.path.join(test_folder_128, 'mask_256')
    os.makedirs(test_folder_256_mask, exist_ok = True)
    """
    train_mask_files = os.listdir(folder_128_mask)

    print(f' Train Folder Start ')

    for i, train_mask_file in enumerate(train_mask_files) :
        train_mask_file_dir = os.path.join(folder_128_mask, train_mask_file)
        train_mask_arr = np.load(train_mask_file_dir)
        trg_size = 256
        trg_mask = np.zeros((trg_size,trg_size))

        padding = (trg_size - 128)//2
        trg_mask[padding:padding+128,padding:padding+128] = train_mask_arr
        np.save(os.path.join(folder_256_mask, train_mask_file), trg_mask)
    """
    print(f' Test Folder Start ')
    test_mask_files = os.listdir(test_folder_128_mask)
    for test_mask_file in test_mask_files :
        test_mask_file_dir = os.path.join(test_folder_128_mask, test_mask_file)
        test_mask_arr = np.load(test_mask_file_dir)
        trg_size = 256
        trg_mask = np.zeros((trg_size,trg_size))
        padding = (trg_size - 128)//2
        trg_mask[padding:padding+128,padding:padding+128] = test_mask_arr
        np.save(os.path.join(test_folder_256_mask, test_mask_file), trg_mask)

if __name__ == '__main__' :
    main()