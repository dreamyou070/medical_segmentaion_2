import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
def main() :

    print(f' step 1. path')
    #base_dir = r'D:\medical\brain\data\BraTS2020_Segmentation_256/TrainingData'
    base_dir = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_256/train/anomal'
    mask_dir = os.path.join(base_dir, 'mask_256')
    mask_files = os.listdir(mask_dir)
    all_values = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask_arr = np.load(mask_path)
        mask_arr = np.where(mask_arr == 4, 3, mask_arr).flatten().tolist()
        # numpy to list
        all_values.extend(mask_arr)
    class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(all_values), y=all_values)
    category = 'brain'
    np.save(f'class_weight_{category}.npy', class_weight)



if __name__ == '__main__':
    main()