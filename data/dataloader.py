import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms
import cv2
from tensorflow.keras.utils import to_categorical
from torch import nn

brain_class_map = {0: ['b', 'brain'],
                   1: ['n', 'non-enhancing tumor core'],
                   2: ['e', 'edema'],
                   3: ['t', 'enhancing tumor'], }
cardiac_class_map = {0: ['b', 'background'],
                     1: ['l', 'left ventricle'],
                     2: ['m', 'myocardium'],
                     3: ['r', 'right ventricle'], }
abdomen_class_map = {0: ['b', 'background'],
                     1: ['x', 'aorta'],
                     2: ['l', 'liver'],
                     3: ['k', 'kidney'],
                     4: ['t', 'tumor'],
                     5: ['v', 'vein'],
                     6: ['s', 'spleen'],
                     7: ['p', 'pancreas'],
                     8: ['g', 'gallbladder'],
                     9: ['f', 'fat'],
                     10: ['m', 'muscle'],
                     11: ['h', 'heart'],
                     12: ['i', 'intestine'],
                     13: ['o', 'other'], }
leader_polyp_class_map = {0: ['b', 'background'],
                          1: ['p', 'non-neoplastic'],
                          2: ['n', 'neoplastic'], }
teeth_class_map = {0: ['b', 'background'],
                   1: ['t', 'anomal']}

# (i) normal- NOR,
# (ii) patients with previous myocardial infarction- MINF,
# (iii) patients with dilated cardiomyopathy- DCM,
# (iv) patients with hypertrophic cardiomyopathy- HCM,
# (v) patients with abnormal right ventricle- ARV

base_prompts = ['this is a picture of ',
                'this is a picture of a ',
                'this is a picture of the ', ]

class_matching_map = {0: np.array([0, 0, 0]),
                      1: np.array([255, 0, 0]),
                      2: np.array([0, 255, 0]),
                      3: np.array([0, 0, 255])}


def passing_mvtec_argument(args):
    global argument
    argument = args


class TestDataset(Dataset):

    def __init__(self,
                 image_root, gt_root,
                 resize_shape=(240, 240),
                 image_processor=None,
                 latent_res: int = 64,
                 n_classes: int = 4,
                 mask_res=128,
                 use_data_aug=False, ):

        # [1] base image
        image_paths, gt_paths = [], []

        rgb_folder = image_root  # anomal / image_256
        gt_folder = gt_root  # [128,128]
        files = os.listdir(rgb_folder)  #
        for file in files:
            name, ext = os.path.splitext(file)
            image_paths.append(os.path.join(rgb_folder, file))
            if argument.gt_ext_npy:
                gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))
            else:
                gt_paths.append(os.path.join(gt_folder, f'{name}{ext}'))

        self.resize_shape = resize_shape
        self.image_processor = image_processor # condition image processor
        self.transform = transforms.Compose([transforms.ToTensor(), # original image processor
                                             transforms.Normalize([0.5], [0.5]), ])
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.latent_res = latent_res
        self.n_classes = n_classes
        self.mask_res = mask_res
        self.use_data_aug = use_data_aug

    def __len__(self):
        return len(self.image_paths)

    def torch_to_pil(self, torch_img):
        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)
        return pil


    def load_image(self, image_path, trg_h, trg_w, type='RGB'):
        image = Image.open(image_path)
        if type == 'RGB':
            if not image.mode == "RGB":
                image = image.convert("RGB")
        elif type == 'L':
            if not image.mode == "L":
                image = image.convert("L")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img




    def __getitem__(self, idx):

        # [1] base
        img_path = self.image_paths[idx]
        pure_path = os.path.split(img_path)[-1]
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1], type='RGB')  # np.array,

        if self.use_data_aug:
            # rotating
            random_p = np.random.rand()
            if random_p < 0.25:
                number = 1
            elif 0.25 <= random_p < 0.5:
                number = 2
            elif 0.5 <= random_p < 0.75:
                number = 3
            elif 0.75 <= random_p:
                number = 4

            img = np.rot90(img, k=number)  # ok, because it is 3 channel image
        img = self.transform(img.copy())

        # [2] gt dir
        gt_path = self.gt_paths[idx]  #
        if argument.gt_ext_npy:
            gt_arr = np.load(gt_path)  # 256,256 (brain tumor case)
            if self.use_data_aug:
                gt_arr = np.rot90(gt_arr, k=number)
            gt_arr = np.where(gt_arr == 4, 3, gt_arr)  # 4 -> 3
            # make image from numpy

            # [1] get final image
            H, W = gt_arr.shape[0], gt_arr.shape[1]
            mask_rgb = np.zeros((H, W, 3))
            for h_index in range(H):
                for w_index in range(W):
                    mask_rgb[h_index, w_index] = class_matching_map[gt_arr[h_index, w_index]]
            mask_pil = Image.fromarray(mask_rgb.astype(np.uint8))
            mask_pil = mask_pil.resize((384, 384), Image.BICUBIC)
            mask_img = self.transform(np.array(mask_pil))


        else:
            gt_img = self.load_image(gt_path, self.mask_res, self.mask_res, type='L')
            gt_arr = np.array(gt_img)  # 128,128
            gt_arr = np.where(gt_arr > 100, 1, 0)

        # [3] make semantic pseudo mal

        # ex) 0, 1, 2
        class_es = np.unique(gt_arr)

        gt_arr_ = to_categorical(gt_arr, num_classes=self.n_classes)
        class_num = gt_arr_.shape[-1]
        gt = np.zeros((self.mask_res,  # 256
                       self.mask_res,  # 256
                       self.n_classes))  # 3

        # 256,256,3
        gt[:, :, :class_num] = gt_arr_
        gt = torch.tensor(gt).permute(2, 0, 1)  # 3,256,256
        # [3] gt flatten
        gt_flat = gt_arr.flatten()  # 128*128

        if argument.use_image_by_caption:

            # [3] caption
            if argument.obj_name == 'brain':
                class_map = brain_class_map
            elif argument.obj_name == 'cardiac':
                class_map = cardiac_class_map
            elif argument.obj_name == 'abdomen':
                class_map = abdomen_class_map
            elif argument.obj_name == 'leader_polyp':
                class_map = leader_polyp_class_map

            if argument.use_base_prompt:
                caption = base_prompts[np.random.randint(0, len(base_prompts))]
            else:
                caption = ''
            caption = base_prompts[np.random.randint(0, len(base_prompts))]
            for i, class_idx in enumerate(class_es):
                if argument.use_key_word:
                    caption += class_map[class_idx][0]
                else:
                    caption += class_map[class_idx][1]

                if i == class_es.shape[0] - 1:
                    caption += ''
                else:
                    # caption += ', '
                    caption += ' '
        else:
            if argument.use_base_prompt:
                base_prompt = base_prompts[np.random.randint(0, len(base_prompts))]
            else:
                base_prompt = ''
            caption = f'{base_prompt}{argument.obj_name}'


        # [3] image pixel

        # condition image = [384,384]
        # gt =[3,256,256]
        gt_pil = gt.permute(1, 2, 0).cpu().numpy() * 255
        gt_pil = gt_pil.astype(np.uint8)
        # remove r channel to zero
        gt_pil[:, :, 0] = gt_pil[:, :, 0] * 0
        gt_pil = Image.fromarray(gt_pil)  # [256,256,3], RGB mask

        if argument.image_processor == 'blip':
            pil = Image.open(img_path).convert('RGB')
            image_condition = self.image_processor(pil)  # [3,224,224]

        elif argument.image_processor == 'pvt':
            pil = Image.open(img_path).convert('RGB')
            image_condition = self.image_processor(pil)

        else:

            image_condition = self.image_processor(images=Image.open(img_path).convert('RGB'),
                                                   return_tensors="pt",
                                                   padding=True)  # .data['pixel_values'] # [1,3,224,224]
            image_condition.data['pixel_values'] = (image_condition.data['pixel_values']).squeeze()
            pixel_value = image_condition.data["pixel_values"]  # [3,224,224]

        # can i use visual attention mask in ViT ?
        return {'image': img,  # [3,512,512]
                "gt": gt,  # [3,256,256]
                "gt_flat": gt_flat,  # [128*128]
                'caption': caption,
                "image_condition": image_condition,
                'pure_path' : pure_path}  # [197,1]


    """
    def __getitem__(self, idx):

        # [1] base
        img_path = self.image_paths[idx]
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1], type='RGB')  # np.array,

        if self.use_data_aug:
            # rotating
            random_p = np.random.rand()
            if random_p < 0.25:
                number = 1
            elif 0.25 <= random_p < 0.5:
                number = 2
            elif 0.5 <= random_p < 0.75:
                number = 3
            elif 0.75 <= random_p:
                number = 4

            img = np.rot90(img, k=number)  # ok, because it is 3 channel image
        img = self.transform(img.copy())

        # [2] gt dir
        gt_path = self.gt_paths[idx]  #
        if argument.gt_ext_npy:
            gt_arr = np.load(gt_path)  # 256,256 (brain tumor case)
            if self.use_data_aug:
                gt_arr = np.rot90(gt_arr, k=number)
            gt_arr = np.where(gt_arr == 4, 3, gt_arr)  # 4 -> 3
            # make image from numpy
            if self.use_data_aug:
                gt_arr = np.rot90(gt_arr, k=number)

            # [1] get final image
            H, W = gt_arr.shape[0], gt_arr.shape[1]
            mask_rgb = np.zeros((H, W, 3))
            for h_index in range(H):
                for w_index in range(W):
                    mask_rgb[h_index, w_index] = class_matching_map[gt_arr[h_index, w_index]]
            mask_pil = Image.fromarray(mask_rgb.astype(np.uint8))
            mask_pil = mask_pil.resize((384, 384), Image.BICUBIC)
            mask_img = self.transform(np.array(mask_pil))


        else:
            gt_img = self.load_image(gt_path, self.mask_res, self.mask_res, type='L')
            if self.use_data_aug:
                gt_arr = np.rot90(gt_img, k=number)
            gt_arr = np.array(gt_img)  # 128,128
            gt_arr = np.where(gt_arr > 100, 1, 0)

        # [3] make semantic pseudo mal

        # ex) 0, 1, 2
        class_es = np.unique(gt_arr)

        gt_arr_ = to_categorical(gt_arr, num_classes=self.n_classes)
        class_num = gt_arr_.shape[-1]
        gt = np.zeros((self.mask_res,  # 256
                       self.mask_res,  # 256
                       self.n_classes))  # 3

        # 256,256,3
        gt[:, :, :class_num] = gt_arr_
        gt = torch.tensor(gt).permute(2, 0, 1)  # 3,256,256
        # [3] gt flatten
        gt_flat = gt_arr.flatten()  # 128*128

        if argument.use_image_by_caption:

            # [3] caption
            if argument.obj_name == 'brain':
                class_map = brain_class_map
            elif argument.obj_name == 'cardiac':
                class_map = cardiac_class_map
            elif argument.obj_name == 'abdomen':
                class_map = abdomen_class_map
            elif argument.obj_name == 'leader_polyp':
                class_map = leader_polyp_class_map

            if argument.use_base_prompt:
                caption = base_prompts[np.random.randint(0, len(base_prompts))]
            else:
                caption = ''
            caption = base_prompts[np.random.randint(0, len(base_prompts))]
            for i, class_idx in enumerate(class_es):
                if argument.use_key_word:
                    caption += class_map[class_idx][0]
                else:
                    caption += class_map[class_idx][1]

                if i == class_es.shape[0] - 1:
                    caption += ''
                else:
                    # caption += ', '
                    caption += ' '
        else:
            if argument.use_base_prompt:
                base_prompt = base_prompts[np.random.randint(0, len(base_prompts))]
            else:
                base_prompt = ''
            caption = f'{base_prompt}{argument.obj_name}'

        caption_token = self.tokenizer(caption,
                                       padding="max_length",
                                       truncation=True, return_tensors="pt")
        input_ids = caption_token.input_ids
        
        # [3] image pixel

        # condition image = [384,384]
        # gt =[3,256,256]
        gt_pil = gt.permute(1, 2, 0).cpu().numpy() * 255
        gt_pil = gt_pil.astype(np.uint8)
        # remove r channel to zero
        gt_pil[:, :, 0] = gt_pil[:, :, 0] * 0
        gt_pil = Image.fromarray(gt_pil)  # [256,256,3], RGB mask

        if argument.image_processor == 'blip':
            pil = Image.open(img_path).convert('RGB')
            image_condition = self.image_processor(pil)  # [3,224,224]

        elif argument.image_processor == 'pvt':
            pil = Image.open(img_path).convert('RGB')
            image_condition = self.image_processor(pil)

        else:

            image_condition = self.image_processor(images=Image.open(img_path).convert('RGB'),
                                                   return_tensors="pt",
                                                   padding=True)  # .data['pixel_values'] # [1,3,224,224]
            image_condition.data['pixel_values'] = (image_condition.data['pixel_values']).squeeze()
            pixel_value = image_condition.data["pixel_values"]  # [3,224,224]

        # can i use visual attention mask in ViT ?
        return {'image': img,  # [3,512,512]
                "gt": gt,  # [3,256,256]
                "gt_flat": gt_flat,  # [128*128]
                "input_ids": input_ids,
                'caption': caption,
                "image_condition": image_condition}  # [197,1]
    """