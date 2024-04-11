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

    folder_128 = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_128'
    folder_256 = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_256'

    folder_128_image = os.path.join(folder_128, 'image_128')
    folder_256_image = os.path.join(folder_256, 'image_256')

    folder_128_mask = os.path.join(folder_128, 'mask_128')
    folder_256_mask = os.path.join(folder_256, 'mask_256')

    test_folder_128 = os.path.join(folder_128, 'test')
    test_folder_256 = os.path.join(folder_256, 'test')
    os.makedirs(test_folder_128, exist_ok=True)
    os.makedirs(test_folder_256, exist_ok=True)

    test_folder_128_image = os.path.join(test_folder_128, 'image_128')
    test_folder_256_image = os.path.join(test_folder_256, 'image_256')
    os.makedirs(test_folder_128_image, exist_ok=True)
    os.makedirs(test_folder_256_image, exist_ok=True)

    test_folder_128_mask = os.path.join(test_folder_128, 'mask_128')
    test_folder_256_mask = os.path.join(test_folder_256, 'mask_256')
    os.makedirs(test_folder_128_mask, exist_ok=True)
    os.makedirs(test_folder_256_mask, exist_ok=True)

    images = os.listdir(folder_128_image)
    for i, image in enumerate(images) :
        if i < len(images) * 0.2 :
            name = image.split('.')[0]

            # [1] 128 image
            image_128_path = os.path.join(folder_128_image, image)
            image_256_path = os.path.join(folder_256_image, image)

            # [2] mask
            mask_128_path = os.path.join(folder_128_mask, f'{name}.npy')
            mask_256_path = os.path.join(folder_256_mask, f'{name}.npy')

            # [3] save
            test_image_128_path = os.path.join(test_folder_128_image, image)
            test_image_256_path = os.path.join(test_folder_256_image, image)
            test_mask_128_path = os.path.join(test_folder_128_mask, f'{name}.npy')
            test_mask_256_path = os.path.join(test_folder_256_mask, f'{name}.npy')

            os.rename(image_128_path, test_image_128_path)
            os.rename(image_256_path, test_image_256_path)
            os.rename(mask_128_path, test_mask_128_path)
            os.rename(mask_256_path, test_mask_256_path)



if __name__ == '__main__' :
    main()