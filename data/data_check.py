import os
import numpy as np

base_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_128'
phases = os.listdir(base_folder)
for phase in phases :
    phase_dir = os.path.join(base_folder, phase)
    anomal_folder = os.path.join(phase_dir, 'anomal')
    image_folder = os.path.join(anomal_folder, 'image_128')
    mask_folder = os.path.join(anomal_folder, 'mask_128')
    print(f'phase : {phase} image num = {len(os.listdir(image_folder))} mask num = {len(os.listdir(mask_folder))}')
    mask_files = os.listdir(mask_folder)
    for mask_file in mask_files :
        mask_path = os.path.join(mask_folder, mask_file)
        mask_arr = np.load(mask_path)
        mask_arr = np.where(mask_arr == 4, 3, mask_arr).flatten().tolist()
        np.save(mask_path, mask_arr)

# [2] 256 data
base_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_256'
phases = os.listdir(base_folder)
for phase in phases :
    phase_dir = os.path.join(base_folder, phase)
    anomal_folder = os.path.join(phase_dir, 'anomal')
    image_folder = os.path.join(anomal_folder, 'image_256')
    mask_folder = os.path.join(anomal_folder, 'mask_256')
    mask_files = os.listdir(mask_folder)

    mask_128_folder = os.path.join(anomal_folder, 'mask_128')
    os.makedirs(mask_128_folder, exist_ok=True)
    for mask_file in mask_files :
        mask_path = os.path.join(mask_folder, mask_file)
        mask_arr = np.load(mask_path)
        mask_arr = np.where(mask_arr == 4, 3, mask_arr).flatten().tolist()
        np.save(mask_path, mask_arr)

        # save 128 after cropping
        target_size = 128
        h, w = mask_arr.shape
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        mask_128 = mask_arr[start_h:start_h+target_size, start_w:start_w+target_size]
        mask_128_path = os.path.join(mask_128_folder, mask_file)
        np.save(mask_128_path, mask_128)