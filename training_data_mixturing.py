import os
from PIL import Image

train_img_folder = '/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/Pranet/train_org/images'
train_gt_folder = '/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/Pranet/train_org/masks'
save_base_dir = '/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/Pranet/train_org/merged'
os.makedirs(save_base_dir, exist_ok=True)
images = os.listdir(train_img_folder)
r = 256
for image in images:

    image_path = os.path.join(train_img_folder, image)
    gt_path = os.path.join(train_gt_folder, image)

    original_pil = Image.open(image_path).resize((r,r)).convert('RGB')
    gt_pil = Image.open(gt_path).resize((r,r)).convert('RGB')

    total_img = Image.new('RGB', (r * 2, r))
    total_img.paste(original_pil, (0, 0))
    total_img.paste(gt_pil, (r, 0))

    total_img.save(os.path.join(save_base_dir, image))