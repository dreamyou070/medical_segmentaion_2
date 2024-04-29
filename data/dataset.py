import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from tensorflow.keras.utils import to_categorical

def passing_mvtec_argument(args):
    global argument
    argument = args

class TrainDataset(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(240, 240),
                 image_processor=None,
                 latent_res: int = 64,
                 n_classes: int = 4,
                 mask_res = 128,
                 use_data_aug = False,):

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths = [], []
        folders = os.listdir(self.root_dir) # anomal
        for folder in folders :
            folder_dir = os.path.join(self.root_dir, folder) # anomal
            folder_res = folder.split('_')[-1]
            rgb_folder = os.path.join(folder_dir, f'image_{folder_res}') # anomal / image_256
            gt_folder = os.path.join(folder_dir, f'mask_{mask_res}')    # [128,128]
            files = os.listdir(rgb_folder) #
            for file in files:
                name, ext = os.path.splitext(file)
                image_paths.append(os.path.join(rgb_folder, file))
                if argument.gt_ext_npy :
                    gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))
                else :
                    gt_paths.append(os.path.join(gt_folder, f'{name}{ext}'))

        self.resize_shape = resize_shape
        self.image_processor = image_processor
        self.transform = transforms.Compose([transforms.ToTensor(),
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
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1], type='RGB')  # np.array,

        if self.use_data_aug :
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
            img = np.rot90(img, k=number) # ok, because it is 3 channel image
        img = self.transform(img.copy())


        # [2] gt dir
        gt_path = self.gt_paths[idx]  #
        try :
            gt_img = self.load_image(gt_path, self.mask_res, self.mask_res, type='L')
        except :
            name, ext = os.path.splitext(gt_path)
            gt_path = f'{name}.png'
            gt_img = self.load_image(gt_path, self.mask_res, self.mask_res, type='L')
        if self.use_data_aug :
            gt_img = np.rot90(gt_img, k=number)
        gt_arr = np.array(gt_img) # 128,128
        gt_arr = np.where(gt_arr > 100, 1, 0)

        # [3] generate res 64 gt image
        gt_64_pil = Image.open(gt_path).convert('L').resize((self.latent_res, self.latent_res), Image.BICUBIC)
        gt_64_array = np.array(gt_64_pil) # [64,64]
        gt_64_array = np.where(gt_64_array > 100, 1, 0) # [64,64]
        gt_64_flat = gt_64_array.flatten()  # 64*64
        gt_64_array = torch.tensor(to_categorical(gt_64_array, num_classes=self.n_classes)).permute(2,0,1).contiguous() # [3,64,64]

        # [3.2]
        gt_32_pil = Image.open(gt_path).convert('L').resize((32, 32), Image.BICUBIC)
        gt_32_array = np.array(gt_32_pil) # [32,32]
        gt_32_array = np.where(gt_32_array > 100, 1, 0)
        gt_32_array = to_categorical(gt_32_array, num_classes=self.n_classes)
        gt_32_array = torch.tensor(gt_32_array).permute(2,0,1).contiguous() # [3,32,32]

        # [3.3]
        gt_16_pil = Image.open(gt_path).convert('L').resize((16, 16), Image.BICUBIC)
        gt_16_array = np.array(gt_16_pil) # [16,16]
        gt_16_array = np.where(gt_16_array > 100, 1, 0)
        gt_16_array = to_categorical(gt_16_array, num_classes=self.n_classes)
        gt_16_array = torch.tensor(gt_16_array).permute(2,0,1).contiguous()

        res_array_gt = {}
        res_array_gt['64'] = gt_64_array
        res_array_gt['32'] = gt_32_array
        res_array_gt['16'] = gt_16_array

        gt_arr_ = to_categorical(gt_arr, num_classes=self.n_classes)
        class_num = gt_arr_.shape[-1]
        gt = np.zeros((self.mask_res,   # 256
                       self.mask_res,   # 256
                       self.n_classes)) # 3

        # 256,256,3
        gt[:,:,:class_num] = gt_arr_
        gt = torch.tensor(gt).permute(2,0,1)        # 3,256,256
        # [3] gt flatten
        gt_flat = gt_arr.flatten() # 128*128

        gt_pil = gt.permute(1,2,0).cpu().numpy() * 255
        gt_pil = gt_pil.astype(np.uint8)
        # remove r channel to zero
        gt_pil[:,:,0] = gt_pil[:,:,0] * 0
        gt_pil = Image.fromarray(gt_pil)  # [256,256,3], RGB mask


        if argument.image_processor == 'blip' :
            pil = Image.open(img_path).convert('RGB')
            if self.use_data_aug :
                np_pil = np.rot90(np.array(pil), k=number)
                pil = Image.fromarray(np_pil)
            image_condition = self.image_processor(pil)  # [3,224,224]

        elif argument.image_processor == 'pvt' :
            pil = Image.open(img_path).convert('RGB')
            if self.use_data_aug :
                np_pil = np.rot90(np.array(pil), k=number)
                pil = Image.fromarray(np_pil)
            image_condition = self.image_processor(pil)

        else :
            if self.use_data_aug :
                pil = Image.open(img_path).convert('RGB')
                np_pil = np.rot90(np.array(pil), k=number)
                image_condition = self.image_processor(images=Image.fromarray(np_pil),
                                                       return_tensors="pt",
                                                       padding=True)  # .data['pixel_values'] # [1,3,224,224]
            else :
                image_condition = self.image_processor(images=Image.open(img_path).convert('RGB'),
                                                       return_tensors="pt",
                                                       padding=True)  # .data['pixel_values'] # [1,3,224,224]
            image_condition.data['pixel_values'] = (image_condition.data['pixel_values']).squeeze()

        # can i use visual attention mask in ViT ?
        return {'image': img,  # [3,512,512]
                "gt": gt,                       # [3,256,256]
                "gt_flat" : gt_flat,            # [128*128]
                "image_condition" : image_condition,
                'res_array_gt' : res_array_gt,
                'gt_64_flat' : gt_64_flat} # [197,1]


class TestDataset(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(240, 240),
                 image_processor=None,
                 latent_res: int = 64,
                 n_classes: int = 4,
                 mask_res=128,
                 use_data_aug=False, ):

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths = [], []

        rgb_folder = os.path.join(self.root_dir, f'images')  # anomal / image_256
        gt_folder = os.path.join(self.root_dir, f'masks')  # [128,128]
        files = os.listdir(rgb_folder)  #
        for file in files:
            name, ext = os.path.splitext(file)
            image_paths.append(os.path.join(rgb_folder, file))
            if argument.gt_ext_npy:
                gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))
            else:
                gt_paths.append(os.path.join(gt_folder, f'{name}{ext}'))

        self.resize_shape = resize_shape
        self.image_processor = image_processor
        self.transform = transforms.Compose([transforms.ToTensor(),
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

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

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
            if argument.obj_name == 'brain':
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
            try:
                gt_img = self.load_image(gt_path, self.mask_res, self.mask_res, type='L')
            except:
                name, ext = os.path.splitext(gt_path)
                gt_path = f'{name}.png'
                gt_img = self.load_image(gt_path, self.mask_res, self.mask_res, type='L')
            if self.use_data_aug:
                gt_img = np.rot90(gt_img, k=number)
            gt_arr = np.array(gt_img)  # 128,128
            gt_arr = np.where(gt_arr > 100, 1, 0)

        # [3] generate res 64 gt image
        gt_64_pil = Image.open(gt_path).convert('L').resize((self.latent_res, self.latent_res), Image.BICUBIC)
        gt_64_array = np.array(gt_64_pil)  # [64,64]
        gt_64_array = np.where(gt_64_array > 100, 1, 0)  # [64,64]
        gt_64_flat = gt_64_array.flatten()  # 64*64
        gt_64_array = torch.tensor(to_categorical(gt_64_array, num_classes=self.n_classes)).permute(2, 0,
                                                                                                    1).contiguous()  # [3,64,64]

        #
        gt_32_pil = Image.open(gt_path).convert('L').resize((32, 32), Image.BICUBIC)
        gt_32_array = np.array(gt_32_pil)  # [32,32]
        gt_32_array = np.where(gt_32_array > 100, 1, 0)
        gt_32_array = to_categorical(gt_32_array, num_classes=self.n_classes)
        gt_32_array = torch.tensor(gt_32_array).permute(2, 0, 1).contiguous()  # [3,32,32]

        gt_16_pil = Image.open(gt_path).convert('L').resize((16, 16), Image.BICUBIC)
        gt_16_array = np.array(gt_16_pil)  # [16,16]
        gt_16_array = np.where(gt_16_array > 100, 1, 0)
        gt_16_array = to_categorical(gt_16_array, num_classes=self.n_classes)
        gt_16_array = torch.tensor(gt_16_array).permute(2, 0, 1).contiguous()

        res_array_gt = {}
        res_array_gt['64'] = gt_64_array
        res_array_gt['32'] = gt_32_array
        res_array_gt['16'] = gt_16_array

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


        # gt =[3,256,256]
        gt_pil = gt.permute(1, 2, 0).cpu().numpy() * 255
        gt_pil = gt_pil.astype(np.uint8)
        # remove r channel to zero
        gt_pil[:, :, 0] = gt_pil[:, :, 0] * 0
        gt_pil = Image.fromarray(gt_pil)  # [256,256,3], RGB mask

        if argument.image_processor == 'blip':
            pil = Image.open(img_path).convert('RGB')
            if self.use_data_aug:
                np_pil = np.rot90(np.array(pil), k=number)
                pil = Image.fromarray(np_pil)
            image_condition = self.image_processor(pil)  # [3,224,224]

        elif argument.image_processor == 'pvt':
            pil = Image.open(img_path).convert('RGB')
            if self.use_data_aug:
                np_pil = np.rot90(np.array(pil), k=number)
                pil = Image.fromarray(np_pil)
            image_condition = self.image_processor(pil)

        else:
            if self.use_data_aug:
                pil = Image.open(img_path).convert('RGB')
                np_pil = np.rot90(np.array(pil), k=number)
                image_condition = self.image_processor(images=Image.fromarray(np_pil),
                                                       return_tensors="pt",
                                                       padding=True)  # .data['pixel_values'] # [1,3,224,224]
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
                "image_condition": image_condition,
                'res_array_gt': res_array_gt,
                'gt_64_flat': gt_64_flat}  # [197,1]