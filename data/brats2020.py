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

    save_folder_128 = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_128'
    os.makedirs(save_folder_128, exist_ok = True)
    save_folder_256 = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_256'
    os.makedirs(save_folder_256, exist_ok=True)

    save_folder_128_image = os.path.join(save_folder_128, 'image_128')
    save_folder_256_image = os.path.join(save_folder_256, 'image_256')
    os.makedirs(save_folder_128_image, exist_ok=True)
    os.makedirs(save_folder_256_image, exist_ok=True)

    save_folder_128_image_sub = os.path.join(save_folder_128, 'image_128_sub')
    save_folder_256_image_sub = os.path.join(save_folder_256, 'image_256_sub')
    os.makedirs(save_folder_128_image_sub, exist_ok=True)
    os.makedirs(save_folder_256_image_sub, exist_ok=True)

    save_folder_128_mask = os.path.join(save_folder_128, 'mask_128')
    save_folder_256_mask = os.path.join(save_folder_256, 'mask_256')
    os.makedirs(save_folder_128_mask, exist_ok=True)
    os.makedirs(save_folder_256_mask, exist_ok=True)

    save_folder_128_mask_sub = os.path.join(save_folder_128, 'mask_128_sub')
    save_folder_256_mask_sub = os.path.join(save_folder_256, 'mask_256_sub')
    os.makedirs(save_folder_128_mask_sub, exist_ok=True)
    os.makedirs(save_folder_256_mask_sub, exist_ok=True)

    main_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation'
    phases = os.listdir(main_folder)
    for phase in phases :
        if 'Train' in phase :
            save_phase_128 = os.path.join(save_folder_128, 'train')
            save_phase_256 = os.path.join(save_folder_256, 'train')
        #else :
        #    save_phase_128 = os.path.join(save_folder_128, 'test')
        #    save_phase_256 = os.path.join(save_folder_256, 'test')
            os.makedirs(save_phase_128, exist_ok = True)
            os.makedirs(save_phase_256, exist_ok=True)
            phase_dir = os.path.join(main_folder, phase)
            phase_dir = os.path.join(phase_dir, f'MICCAI_{phase}')
            folders = os.listdir(phase_dir)
            for folder in folders :
                folder_dir = os.path.join(phase_dir, folder)

                # image file
                flair_file_dir = os.path.join(folder_dir, f'{folder}_flair.nii')
                t1_file_dir = os.path.join(folder_dir, f'{folder}_t1.nii')
                t1ce_file_dir = os.path.join(folder_dir, f'{folder}_t1ce.nii')
                t2_file_dir = os.path.join(folder_dir, f'{folder}_t2.nii')

                # get image
                flair_img = nib.load(flair_file_dir).get_fdata() # 240, 240, 155
                t1_img = nib.load(t1_file_dir).get_fdata()       # 240, 240, 155
                t1ce_img = nib.load(t1ce_file_dir).get_fdata()   # 240, 240, 155
                t2_img = nib.load(t2_file_dir).get_fdata()       # 240, 240, 155

                # 3D image
                h,w,c = flair_img.shape
                img = np.zeros((h,w,c,3))
                img[:,:,:,0] = flair_img
                img[:,:,:,1] = t1ce_img
                img[:,:,:,2] = t2_img # [240,240,155,3]
                img = img[:,:,13:141,:]

                # scaling image
                scaler = MinMaxScaler()
                img = scaling_img(img, scaler)

                # mask
                seg_dir = os.path.join(folder_dir, f'{folder}_seg.nii')
                mask_img = nib.load(seg_dir).get_fdata()                 # (240, 240, 155)
                mask = mask_img[:,:,13:141]                    # (128, 128, 128)
                mask = np.where(mask == 4, 3, mask)

                all_axis = mask.shape[-1]
                for i in range(all_axis) :
                    # [1] 128 image
                    img_128_arr = img[56:184,56:184,i,:]
                    img_240_arr = img[:,:,i,:]
                    mask_128_arr = mask[56:184,56:184,i]
                    val, counts = np.unique(mask_128_arr, return_counts=True)
                    mask_240_arr = mask[:, :, i]
                    # [2] padding to make 256 image
                    img_256_arr = np.zeros((256, 256, 3))
                    mask_256_arr = np.zeros((256, 256))
                    pad = (256 - 240) // 2
                    img_256_arr[pad:pad + 240, pad:pad + 240, :] = img_240_arr
                    mask_256_arr[pad:pad + 240, pad:pad + 240] = mask_240_arr

                    if (1 - counts[0] / counts.sum()) > 0.1 :

                        # [3] save image
                        img_128_pil = Image.fromarray((img_128_arr * 255).astype(np.uint8))
                        img_128_pil.save(os.path.join(save_folder_128_image, f'{folder}_{i}.jpg'))
                        img_256_pil = Image.fromarray((img_256_arr * 255).astype(np.uint8))
                        img_256_pil.save(os.path.join(save_folder_256_image, f'{folder}_{i}.jpg'))
                        # [4] save mask
                        np.save(os.path.join(save_folder_128_mask, f'{folder}_{i}.npy'), mask_128_arr)
                        np.save(os.path.join(save_folder_256_mask, f'{folder}_{i}.npy'), mask_256_arr)

                    else :

                        img_128_pil = Image.fromarray((img_128_arr * 255).astype(np.uint8))
                        img_128_pil.save(os.path.join(save_folder_128_image_sub, f'{folder}_{i}.jpg'))
                        img_256_pil = Image.fromarray((img_256_arr * 255).astype(np.uint8))
                        img_256_pil.save(os.path.join(save_folder_256_image_sub, f'{folder}_{i}.jpg'))
                        # [4] save mask
                        np.save(os.path.join(save_folder_128_mask_sub, f'{folder}_{i}.npy'), mask_128_arr)
                        np.save(os.path.join(save_folder_256_mask_sub, f'{folder}_{i}.npy'), mask_256_arr)


if __name__ == '__main__' :
    main()