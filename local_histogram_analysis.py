import os
from PIL import Image
import numpy as np
import cv2
from image_enhancement import *
base_folder = r'C:\Users\hpuser\Desktop\TrainDataset\TrainDataset'
theta0 = 1.3945
theta1 = 9.1377
theta2 = 0.5

img_folder = os.path.join(base_folder, 'images')
mask_folder = os.path.join(base_folder, 'masks')
images = os.listdir(img_folder)

class_0_folder = os.path.join(base_folder, 'class_0')
class_1_folder = os.path.join(base_folder, 'class_1')
class_2_folder = os.path.join(base_folder, 'class_2')
os.makedirs(class_0_folder, exist_ok=True)
os.makedirs(class_1_folder, exist_ok=True)
os.makedirs(class_2_folder, exist_ok=True)
class_0_img_folder = os.path.join(class_0_folder, 'images')
class_0_mask_folder = os.path.join(class_0_folder, 'masks')
os.makedirs(class_0_img_folder, exist_ok=True)
os.makedirs(class_0_mask_folder, exist_ok=True)
class_1_img_folder = os.path.join(class_1_folder, 'images')
class_1_mask_folder = os.path.join(class_1_folder, 'masks')
os.makedirs(class_1_img_folder, exist_ok=True)
os.makedirs(class_1_mask_folder, exist_ok=True)
class_2_img_folder = os.path.join(class_2_folder, 'images')
class_2_mask_folder = os.path.join(class_2_folder, 'masks')
os.makedirs(class_2_img_folder, exist_ok=True)
os.makedirs(class_2_mask_folder, exist_ok=True)

for img_file in images :
    img_dir = os.path.join(img_folder, img_file)
    mask_dir = os.path.join(mask_folder, img_file)

    pil_img = Image.open(img_dir).resize((256,256))
    mask_img = Image.open(mask_dir).resize((256,256)).convert('L')
    mask_np = np.array(mask_img)
    mask_np = np.where(mask_np > 100, 1, 0)
    white_position = mask_np.sum()
    white_position = (white_position / (256 * 256)) * 100
    if white_position < 5 :
        pil_img.save(os.path.join(class_0_img_folder, img_file))
        mask_img.save(os.path.join(class_0_mask_folder, img_file))

    elif 5 <= white_position < 15 :
        pil_img.save(os.path.join(class_1_img_folder, img_file))
        mask_img.save(os.path.join(class_1_mask_folder, img_file))

    else :
        pil_img.save(os.path.join(class_2_img_folder, img_file))
        mask_img.save(os.path.join(class_2_mask_folder, img_file))