# [1] train test spliting

import os,shutil

save_base_folder =  r'D:\medical\Abdomen\Abdomen_TransUnet'
os.makedirs(save_base_folder, exist_ok=True)
val_folder = os.path.join(save_base_folder, 'val')
os.makedirs(val_folder, exist_ok=True)
val_img_folder = os.path.join(val_folder, 'img')
os.makedirs(val_img_folder, exist_ok=True)
val_label_folder = os.path.join(val_folder, 'label')
os.makedirs(val_label_folder, exist_ok=True)


train_folder = os.path.join(save_base_folder, 'train')
os.makedirs(train_folder, exist_ok=True)
train_img_folder = os.path.join(train_folder, 'img')
os.makedirs(train_img_folder, exist_ok=True)
train_label_folder = os.path.join(train_folder, 'label')
os.makedirs(train_label_folder, exist_ok=True)

test_list_file = r'lists/lists_Synapse/test_vol.txt'
with open(test_list_file, 'r') as f:
    test_list = f.readlines()
test_list = [i.strip().replace('case','') for i in test_list]
print(test_list)
base_folder = r'D:\medical\Abdomen\Abdomen\RawData\Training\org'
image_folder = os.path.join(base_folder, 'img')
image_files = os.listdir(image_folder)
for image_file in image_files:
    name = image_file.split('.')[0]
    unique_num = name.replace('img','')
    label_org_dir = os.path.join(base_folder, 'label', 'label' + unique_num + '.nii.gz')
    if unique_num in test_list:
        new_dir = os.path.join(val_img_folder, image_file)
        label_new_dir = os.path.join(val_label_folder, 'label'+unique_num+'.nii.gz')
    else:
        new_dir = os.path.join(train_img_folder, image_file)
        label_new_dir = os.path.join(train_label_folder, 'label'+unique_num+'.nii.gz')
    shutil.copy(os.path.join(image_folder, image_file), new_dir)
    shutil.copy(label_org_dir, label_new_dir)
