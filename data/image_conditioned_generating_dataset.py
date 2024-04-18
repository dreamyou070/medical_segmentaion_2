from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms

def passing_mvtec_argument(args):
    global argument
    argument = args

class_matching_map = {0 : np.array([0,0,0]),
                          1 : np.array([255,0,0]),
                          2 : np.array([0,255,0]),
                          3 : np.array([0,0,255])}
class TrainDataset(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(512,512),
                 image_processor=None, # as if tokenizer
                 latent_res: int = 64,
                 n_classes: int = 4,
                 mask_res=128):

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths = [], []
        folders = os.listdir(self.root_dir)  # anomal
        for folder in folders:
            folder_dir = os.path.join(self.root_dir, folder)  # anomal
            folder_res = folder.split('_')[-1]
            rgb_folder = os.path.join(folder_dir, f'image_{folder_res}')  # [conditioned image]
            gt_folder = os.path.join(folder_dir, f'mask_{mask_res}')      # [128,128] (goal of generating model)
            files = os.listdir(rgb_folder)  #
            for file in files:
                name, ext = os.path.splitext(file)
                image_paths.append(os.path.join(rgb_folder, file))
                if argument.gt_ext_npy:
                    gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))
                else:
                    gt_paths.append(os.path.join(gt_folder, f'{name}{ext}'))
        self.image_paths = image_paths
        self.gt_paths = gt_paths

        # [2] final image
        self.resize_shape = resize_shape
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])

        # [3]
        self.condition_img_tokenizer = image_processor # necessary for conditioned image


        self.latent_res = latent_res
        self.n_classes = n_classes
        self.mask_res = mask_res

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
    
    # class matching map
    

    def __getitem__(self, idx):

        # [1] get final image
        mask_path = self.gt_paths[idx]
        mask = np.load(mask_path)      # 256,256
        H, W = mask.shape[0], mask.shape[1]
        # L mode to RGB mode
        mask_rgb = mask_base = np.zeros((H, W, 3))
        for h_index in range(H):
            for w_index in range(W):
                mask_rgb[h_index, w_index] = self.class_matching_map[mask[h_index, w_index]]
        mask_pil = Image.fromarray(mask_rgb.astype(np.uint8))
        # resizing
        mask_pil = mask_pil.resize(self.resize_shape, Image.BICUBIC)
        mask_img = self.transform(np.array(mask_pil))

        # [2] get condition image
        image = Image.open(self.image_paths[idx])
        inputs = self.condition_img_tokenizer(images=image, return_tensors="pt").squeeze(0)

        sample = {'image': mask_img,
                  'condition_image' : inputs}

        return sample


class TestDataset(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(512, 512),
                 image_processor=None,  # as if tokenizer
                 latent_res: int = 64,
                 n_classes: int = 4,
                 mask_res=128):

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths = [], []
        folders = os.listdir(self.root_dir)  # anomal
        for folder in folders:
            folder_dir = os.path.join(self.root_dir, folder)  # anomal
            folder_res = folder.split('_')[-1]
            rgb_folder = os.path.join(folder_dir, f'images')  # anomal / image_256
            gt_folder = os.path.join(folder_dir, f'masks')  # [128,128]
            files = os.listdir(rgb_folder)  #
            for file in files:
                name, ext = os.path.splitext(file)
                image_paths.append(os.path.join(rgb_folder, file))
                if argument.gt_ext_npy:
                    gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))
                else:
                    gt_paths.append(os.path.join(gt_folder, f'{name}{ext}'))
        self.image_paths = image_paths
        self.gt_paths = gt_paths

        # [2] final image
        self.resize_shape = resize_shape
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])

        # [3]
        self.condition_img_tokenizer = image_processor  # necessary for conditioned image

        self.latent_res = latent_res
        self.n_classes = n_classes
        self.mask_res = mask_res

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

    # class matching map

    def __getitem__(self, idx):

        # [1] get final image
        mask_path = self.gt_paths[idx]
        mask = np.load(mask_path)  # 256,256
        H, W = mask.shape[0], mask.shape[1]
        # L mode to RGB mode
        mask_rgb = mask_base = np.zeros((H, W, 3))
        for h_index in range(H):
            for w_index in range(W):
                mask_rgb[h_index, w_index] = self.class_matching_map[mask[h_index, w_index]]
        mask_pil = Image.fromarray(mask_rgb.astype(np.uint8))
        # resizing
        mask_pil = mask_pil.resize(self.resize_shape, Image.BICUBIC)
        mask_img = self.transform(np.array(mask_pil))

        # [2] get condition image
        image = Image.open(self.image_paths[idx])
        inputs = self.condition_img_tokenizer(images=image, return_tensors="pt").squeeze(0)

        sample = {'image': mask_img,
                  'condition_image': inputs}

        return sample

def call_dataset(args) :

    if args.image_processor == 'clip':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif args.image_processor == 'vit':
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # [2] train & test dataset
    train_dataset = TrainDataset(root_dir=args.train_data_path,
                                 resize_shape=[args.resize_shape,args.resize_shape],
                                 image_processor=processor,
                                 latent_res=args.latent_res,
                                 n_classes = args.n_classes,
                                 mask_res = args.mask_res)

    test_dataset = TestDataset(root_dir=args.test_data_path,
                               resize_shape=[args.resize_shape,args.resize_shape],
                               image_processor=processor,
                               latent_res=args.latent_res,
                               n_classes=args.n_classes,
                               mask_res = args.mask_res)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False)
    return train_dataloader, test_dataloader