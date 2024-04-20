import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms
import cv2
from tensorflow.keras.utils import to_categorical

brain_class_map = {0: ['b','brain'],
                   1: ['n','non-enhancing tumor core'],
                   2: ['e','edema'],
                   3: ['t','enhancing tumor'],}
cardiac_class_map = {0: ['b','background'],
                        1: ['l','left ventricle'],
                        2: ['m','myocardium'],
                        3: ['r','right ventricle'],}
abdomen_class_map = {0: ['b','background'],
                        1: ['x','aorta'],
                        2: ['l','liver'],
                        3: ['k','kidney'],
                        4: ['t','tumor'],
                        5: ['v','vein'],
                        6: ['s','spleen'],
                        7: ['p','pancreas'],
                        8: ['g','gallbladder'],
                        9: ['f','fat'],
                        10: ['m','muscle'],
                        11: ['h','heart'],
                        12: ['i','intestine'],
                        13: ['o','other'], }
leader_polyp_class_map = {0: ['b','background'],
                          1: ['p','non-neoplastic'],
                          2: ['n','neoplastic'],}
teeth_class_map = {0: ['b','background'],
                     1: ['t','anomal']}

# (i) normal- NOR,
# (ii) patients with previous myocardial infarction- MINF,
# (iii) patients with dilated cardiomyopathy- DCM,
# (iv) patients with hypertrophic cardiomyopathy- HCM,
# (v) patients with abnormal right ventricle- ARV

base_prompts = ['this is a picture of ',
                'this is a picture of a ',
                'this is a picture of the ',]

class_matching_map = {0 : np.array([0,0,0]),
                          1 : np.array([255,0,0]),
                          2 : np.array([0,255,0]),
                          3 : np.array([0,0,255])}
def passing_mvtec_argument(args):
    global argument
    argument = args


class TrainDataset_Seg(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(240, 240),
                 tokenizer=None,
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
        self.tokenizer = tokenizer
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
        if argument.gt_ext_npy :
            gt_arr = np.load(gt_path)     # 256,256 (brain tumor case)
            if self.use_data_aug:
                gt_arr = np.rot90(gt_arr, k=number)
            gt_arr = np.where(gt_arr==4, 3, gt_arr) # 4 -> 3
            # make image from numpy

            # [1] get final image
            H, W = gt_arr.shape[0], gt_arr.shape[1]
            mask_rgb = np.zeros((H, W, 3))
            for h_index in range(H):
                for w_index in range(W):
                    mask_rgb[h_index, w_index] = class_matching_map[gt_arr[h_index, w_index]]
            mask_pil = Image.fromarray(mask_rgb.astype(np.uint8))
            mask_pil = mask_pil.resize((384,384), Image.BICUBIC)
            mask_img = self.transform(np.array(mask_pil))


        else :
            gt_img = self.load_image(gt_path, self.mask_res, self.mask_res, type='L')
            gt_arr = np.array(gt_img) # 128,128
            gt_arr = np.where(gt_arr > 100, 1, 0)

        # [3] make semantic pseudo mal

        # ex) 0, 1, 2
        class_es = np.unique(gt_arr)

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

        if argument.use_image_by_caption :

            # [3] caption
            if argument.obj_name == 'brain':
                class_map = brain_class_map
            elif argument.obj_name == 'cardiac':
                class_map = cardiac_class_map
            elif argument.obj_name == 'abdomen':
                class_map = abdomen_class_map
            elif argument.obj_name == 'leader_polyp':
                class_map = leader_polyp_class_map

            if argument.use_base_prompt :
                caption = base_prompts[np.random.randint(0, len(base_prompts))]
            else :
                caption = ''
            caption = base_prompts[np.random.randint(0, len(base_prompts))]
            for i, class_idx in enumerate(class_es):
                if argument.use_key_word :
                    caption += class_map[class_idx][0]
                else :
                    caption += class_map[class_idx][1]

                if i == class_es.shape[0] - 1:
                    caption += ''
                else:
                    #caption += ', '
                    caption += ' '
        else :
            if argument.use_base_prompt :
                base_prompt = base_prompts[np.random.randint(0, len(base_prompts))]
            else :
                base_prompt = ''
            caption = f'{base_prompt}{argument.obj_name}'

        caption_token = self.tokenizer(caption,
                                       padding="max_length",
                                       truncation=True, return_tensors="pt")
        input_ids = caption_token.input_ids
        """
        key_words = [class_map[i][0] for i in class_es]  # [b,n,e]

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
            default = [0]  # cls token index
            default.extend(get_target_index(key_words, caption))
            key_word_index = default
        else:
            key_word_index = get_target_index(key_words, caption)
        """
        # [3] image pixel

        # condition image = [384,384]
        # gt =[3,256,256]
        gt_pil = gt.permute(1,2,0).cpu().numpy() * 255
        gt_pil = gt_pil.astype(np.uint8)
        # remove r channel to zero
        gt_pil[:,:,0] = gt_pil[:,:,0] * 0
        gt_pil = Image.fromarray(gt_pil)  # [256,256,3], RGB mask

        gt_arr = np.load(gt_path)  # 256,256 (brain tumor case)
        binary_gt = np.where(gt_arr > 0, 255, 0)
        binary_gt = Image.fromarray(binary_gt.astype(np.uint8))
        binary_gt = binary_gt.resize((224, 224), Image.BICUBIC)  # only object
        binary_gt = np.array(binary_gt)
        binary_gt = np.where(binary_gt > 100, 1, 0)  # binary mask
        # mask_embedding
        kernel_size = 16
        stride = 16
        mask_embedding = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        mask_embedding.weight.data.fill_(1)
        mask_embedding.weight.requires_grad = False
        mask_embedding = mask_embedding(torch.tensor(binary_gt).unsqueeze(0).float())  # [1,14,14]
        # 3dim to 2 dim
        cls_embedding = torch.ones((1, 1))
        mask_embedding = mask_embedding.view(1, -1).contiguous()  # 1, 196
        mask_embedding = torch.cat([cls_embedding, mask_embedding], dim=1).unsqueeze(dim=-1)  # 1, 197,1
        mask_embedding = mask_embedding.squeeze(0)  # 197, 1

        if argument.image_processor == 'blip' :
            pil = Image.open(img_path).convert('RGB')
            image_condition = self.image_processor(pil)  # [3,224,224]
        else :
            image_condition = self.image_processor(images=Image.open(img_path),
                                                    return_tensors="pt",
                                                    padding=True)  # .data['pixel_values'] # [1,3,224,224]
            image_condition.data['pixel_values'] = (image_condition.data['pixel_values']).squeeze()
            pixel_value = image_condition.data["pixel_values"]  # [3,224,224]



        # can i use visual attention mask in ViT ?
        return {'image': img,  # [3,512,512]
                "gt": gt,                       # [3,256,256]
                "gt_flat" : gt_flat,            # [128*128]
                "input_ids": input_ids,
                'caption' : caption,
                "image_condition" : image_condition,
                'mask_embedding' : mask_embedding} # [197,1]

class TestDataset_Seg(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(240, 240),
                 tokenizer=None,
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
            rgb_folder = os.path.join(folder_dir, f'images') # anomal / image_256
            gt_folder = os.path.join(folder_dir, f'masks')    # [128,128]
            files = os.listdir(rgb_folder) #
            for file in files:
                name, ext = os.path.splitext(file)
                image_paths.append(os.path.join(rgb_folder, file))
                if argument.gt_ext_npy :
                    gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))
                else :
                    gt_paths.append(os.path.join(gt_folder, f'{name}{ext}'))

        self.resize_shape = resize_shape
        self.tokenizer = tokenizer
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
            gt_arr = np.where(gt_arr == 4, 3, gt_arr)  # 4 -> 3
        else:
            gt_img = self.load_image(gt_path, self.mask_res, self.mask_res, type='L')
            gt_arr = np.array(gt_img)  # 128,128
            gt_arr = np.where(gt_arr > 100, 1, 0)

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
                if argument.use_key_word :
                    caption += class_map[class_idx][0]
                else :
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
        """
        key_words = [class_map[i][0] for i in class_es]  # [b,n,e]

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
            default = [0]  # cls token index
            default.extend(get_target_index(key_words, caption))
            key_word_index = default
        else:
            key_word_index = get_target_index(key_words, caption)
        """
        # [3] image pixel

        if argument.image_processor == 'blip':
            pil = Image.open(img_path).convert('RGB')
            image_condition = self.image_processor(pil)

        else:
            image_condition = self.image_processor(images=Image.open(img_path),
                                                   return_tensors="pt",
                                                   padding=True)  # .data['pixel_values'] # [1,3,224,224]
            image_condition.data['pixel_values'] = (image_condition.data['pixel_values']).squeeze()
            pixel_value = image_condition.data["pixel_values"]  # [3,224,224]

        return {'image': img,  # [3,512,512]
                "gt": gt,  # [3,256,256]
                "gt_flat": gt_flat,  # [128*128]
                "input_ids": input_ids,
                'caption': caption,
                "image_condition": image_condition}  # [0,3,4]