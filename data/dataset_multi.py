import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms
import cv2
from tensorflow.keras.utils import to_categorical

class_map = {0: ['b','brain'],
             1: ['n','non-enhancing tumor core'],
             2: ['e','edema'],
             3: ['t','enhancing tumor'],}

base_prompts = ['this is a picture of ',
                'this is a picture of a ',
                'this is a picture of an ',
                'this is a picture of the ',
                'this picture is of ',]

def passing_mvtec_argument(args):
    global argument

    argument = args


class TestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx}

        return sample


class TrainDataset_Seg(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(240, 240),
                 tokenizer=None,
                 latent_res: int = 64,
                 n_classes: int = 4,
                 mask_res = 128,
                 use_data_aug = False,):

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths = [], []
        folders = os.listdir(self.root_dir)
        for folder in folders :
            folder_dir = os.path.join(self.root_dir, folder)
            folder_res = folder.split('_')[-1]
            rgb_folder = os.path.join(folder_dir, f'image_{folder_res}')
            gt_folder = os.path.join(folder_dir, f'mask_{mask_res}') # [128,128]
            files = os.listdir(rgb_folder)
            for file in files:
                name, ext = os.path.splitext(file)
                image_paths.append(os.path.join(rgb_folder, file))
                gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))

        self.resize_shape = resize_shape
        self.tokenizer = tokenizer
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
        gt_arr = np.load(gt_path)     # 256,256 (brain tumor case)
        if self.use_data_aug:
            gt_arr = np.rot90(gt_arr, k=number)
        gt_arr = np.where(gt_arr==4, 3, gt_arr) # 4 -> 3

        class_es = np.unique(gt_arr) # key_words

        if argument.binary_test :
            gt_arr = np.where(gt_arr==1, 1, 0)

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

        # [3] caption
        caption = base_prompts[np.random.randint(0, len(base_prompts))]
        for i, class_idx in enumerate(class_es):
            caption += class_map[class_idx][0]
            if i < len(class_es) - 1:
                caption += ', '
        if argument.use_cls_token:
            key_words = [class_map[i][0] for i in class_es if i != 0] # [n,e]
        else :
            key_words = [class_map[i][0] for i in class_es] # [b,n,e]
        # final caption = 'this is a picture of b, n, e, t'
        caption_token = self.tokenizer(caption, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = caption_token.input_ids
        attention_mask = caption_token.attention_mask

        def get_target_index(target_words, caption):

            target_word_index = []
            for target_word in target_words:
                target_word_token = self.tokenizer(target_word, return_tensors="pt")
                target_word_input_ids = target_word_token.input_ids[:, 1]

                # [1] get target word index from sentence token
                sentence_token = self.tokenizer(caption, return_tensors="pt")
                sentence_token = sentence_token.input_ids
                batch_size = sentence_token.size(0)

                for i in range(batch_size):
                    # same number from sentence token to target_word_inpud_ids
                    s_tokens = sentence_token[i]
                    idx = (torch.where(s_tokens == target_word_input_ids))[0].item()
                    target_word_index.append(idx)
            return target_word_index

        if argument.use_cls_token:
            default = [0] # cls token index
            default.extend(get_target_index(key_words, caption))
            key_word_index = default
        else :
            key_word_index = get_target_index(key_words, caption)

        return {'image': img,  # [3,512,512]
                "gt": gt,                       # [3,256,256]
                "gt_flat" : gt_flat,            # [128*128]
                "input_ids": input_ids,
                "key_word_index":key_word_index} # [0,3,4]