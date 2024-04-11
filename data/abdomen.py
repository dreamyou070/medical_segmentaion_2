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


    base_folder = r'D:\medical\Abdomen\Abdomen\RawData\Training\sy/abdomen_512'


    image_512_sub_folder = os.path.join(base_folder, 'image_512_sub')
    mask_512_sub_folder = os.path.join(base_folder, 'mask_512_sub')
    os.makedirs(image_512_sub_folder, exist_ok=True)
    os.makedirs(mask_512_sub_folder, exist_ok=True)

    save_folder_256 = r'D:\medical\Abdomen\Abdomen\RawData\Training\sy/abdomen_256'
    os.makedirs(save_folder_256, exist_ok=True)
    image_256_sub_folder = os.path.join(save_folder_256, 'image_256_sub')
    mask_256_sub_folder = os.path.join(save_folder_256, 'mask_256_sub')
    os.makedirs(image_256_sub_folder, exist_ok=True)
    os.makedirs(mask_256_sub_folder, exist_ok=True)

    phases = ['train',  'test']
    for phase in phases :

        phase_dir = os.path.join(base_folder, phase)
        save_phase_256 = os.path.join(save_folder_256, phase)
        os.makedirs(save_phase_256, exist_ok=True)

        anomal_folder = os.path.join(phase_dir, 'anomal')
        save_phase_256_anomal = os.path.join(save_phase_256, 'anomal')
        os.makedirs(save_phase_256_anomal, exist_ok=True)

        image_folder = os.path.join(anomal_folder, 'image_512')
        save_image_folder_256 = os.path.join(save_phase_256_anomal, 'image_256')
        os.makedirs(save_image_folder_256, exist_ok=True)

        mask_folder = os.path.join(anomal_folder, 'mask_512')
        save_mask_folder_256 = os.path.join(save_phase_256_anomal, 'mask_256')
        os.makedirs(save_mask_folder_256, exist_ok=True)



        images = os.listdir(image_folder)

        for image in images :
            name = os.path.splitext(image)[0]
            # [1] image
            image_path = os.path.join(image_folder, image)

            # [2] mask
            mask_path = os.path.join(mask_folder, f'{name}.npy')
            mask_arr = np.load(mask_path)
            val, counts = np.unique(mask_arr, return_counts=True)

            # [3] 256 image
            image_arr = np.array(Image.open(image_path))
            crop_256 = (512 - 256) // 2
            image_256_arr = image_arr[crop_256:crop_256+256,crop_256:crop_256+256]
            mask_256_arr = mask_arr[crop_256:crop_256+256,crop_256:crop_256+256]
            image_256_pil = Image.fromarray(image_256_arr)

            if (1 - counts[0] / counts.sum()) > 0.1 :
                # [4] save image
                image_256_pil.save(os.path.join(save_image_folder_256, image))
                # [5] save mask
                np.save(os.path.join(save_mask_folder_256, f'{name}.npy'), mask_256_arr)
            else :
                # rename
                os.rename(image_path, os.path.join(image_512_sub_folder, image))
                os.rename(mask_path, os.path.join(mask_512_sub_folder, f'{name}.npy'))

                image_256_pil.save(os.path.join(save_image_folder_256, image))
                # [5] save mask
                np.save(os.path.join(save_mask_folder_256, f'{name}.npy'), mask_256_arr)




if __name__ == '__main__' :
    main()