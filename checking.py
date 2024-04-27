import os

#base_folder = r'/home/dreamyou070/MyData/anomaly_detection/camouflaged/CAMO-V.1.0-CVIU2019_sy/CAMO-V.1.0-CVIU2019_sy/train/res_256'
#image_folder = os.path.join(base_folder, 'image_256')
#gt_folder = os.path.join(base_folder, 'mask_256')

#image_list = os.listdir(image_folder)
#gt_list = os.listdir(gt_folder)
#for img in image_list :
#    img_dir = os.path.join(image_folder, img)
#    gt_dir = os.path.join(gt_folder, img)

gt_path = '/home/dreamyou070/MyData/anomaly_detection/camouflaged/CAMO-V.1.0-CVIU2019_sy/CAMO-V.1.0-CVIU2019_sy/train/res_256/3.jpg'
name, ext = os.path.splitext(gt_path)
print(name)