import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss

from monai.inferers import sliding_window_inference
from monai.transforms import (AsDiscrete, EnsureChannelFirstd, Compose, CropForegroundd, LoadImaged,
                              Orientationd, RandFlipd, RandCropByPosNegLabeld, RandShiftIntensityd,
                              ScaleIntensityRanged, Spacingd, RandRotate90d, )
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch, )
import torch
from PIL import Image
import numpy as np

def main():
    print(f' step 1. set data directory')
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory  # saving directory

    print(f' step 2. Setup transforms for training and validation')
    val_transforms = Compose([LoadImaged(keys=["image", "label"]),
                              EnsureChannelFirstd(keys=["image", "label"]),
                              Orientationd(keys=["image", "label"], axcodes="RAS"), # {image: [batch, 512,512,axis] }
                              Spacingd(keys=["image", "label"],pixdim=(1.5, 1.5, 2.0),mode=("bilinear", "nearest"),),
                              ScaleIntensityRanged(keys=["image"],
                                                   a_min=-175, a_max=250, b_min=0.0, b_max=1.0,
                                                   clip=True),
                              CropForegroundd(keys=["image", "label"], source_key="image"),])
    data_dir = r"/share0/dreamyou070/dreamyou070/MultiSegmentation/result/medical/data"
    #data_dir = r"data"
    split_json = "dataset_1.json"
    datasets = os.path.join(data_dir, split_json)
    val_files = load_decathlon_datalist(datasets,
                                        True,
                                        "validation")
    val_ds = CacheDataset(data=val_files,
                          transform=val_transforms,
                          cache_num=6, cache_rate=1.0, num_workers=4)
    #val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f' step 3. Load the model')
    print(f' len of val_ds = {len(val_ds)}')

    saving_base_dir = '/home/dreamyou070/MyData/anomaly_detection/medical/abdomen/abdomen_re'
    if not os.path.exists(saving_base_dir):
        os.makedirs(saving_base_dir)
    saving_img_dir = os.path.join(saving_base_dir, 'image_512')
    if not os.path.exists(saving_img_dir):
        os.makedirs(saving_img_dir)
    saving_label_dir = os.path.join(saving_base_dir, 'mask_512')
    if not os.path.exists(saving_label_dir):
        os.makedirs(saving_label_dir)
    global_size = 512
    for j in range(len(val_ds)):
        sample = val_ds.__getitem__(j)
        image = sample["image"] # arr
        label = sample["label"] # arr
        axis = image.shape[-1]

        for i in range(axis):
            img = image[...,i].squeeze()
            mask = label[...,i].squeeze()
            h,w = img.shape
            if h < global_size or w < global_size:
                # padding
                background = torch.zeros((global_size,global_size))
                # center padding
                pad_w = global_size - w
                pad_h = global_size - h
                pad_w1 = pad_w // 2
                pad_h1 = pad_h // 2
                background[pad_h1:pad_h1+h,pad_w1:pad_w1+w] = img
                # mask padding
                mask_background = torch.zeros((global_size,global_size))
                mask_background[pad_h1:pad_h1+h,pad_w1:pad_w1+w] = mask
                img = background.numpy()
                img = np.rot90(img, k=1)
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                mask = mask_background.numpy()
                mask = np.rot90(mask, k=1)
                pil_img.save(os.path.join(saving_img_dir, f'sample_{j}_{i}.jpg'))
                np.save(os.path.join(saving_label_dir, f'sample_{j}_{i}.npy'), mask)



if __name__ == '__main__':
    main()
