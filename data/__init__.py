import os
import torch
from model.tokenizer import load_tokenizer
from data.dataset import TrainDataset, TestDataset
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def call_dataset(args) :


    if args.image_processor == 'clip':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    elif args.image_processor == 'vit':
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    elif args.image_processor == 'pvt' :
        processor = transforms.Compose([transforms.Resize((384,384)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # [2] train & test dataset
    train_dataset = TrainDataset(root_dir=args.train_data_path,
                                 resize_shape=[args.resize_shape,args.resize_shape],
                                 image_processor=processor,
                                 latent_res=args.latent_res,
                                 n_classes = args.n_classes,
                                 mask_res = args.mask_res,
                                 use_data_aug = args.use_data_aug,)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

    return train_dataloader

def call_test_dataset(args, _data_name) :

    # [1] data_path here

    base_path = args.base_path
    test_base_path = os.path.join(base_path, 'test')
    data_path = os.path.join(test_base_path, _data_name)

    if args.image_processor == 'clip':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    elif args.image_processor == 'vit':
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    elif args.image_processor == 'pvt' :
        processor = transforms.Compose([transforms.Resize((384,384)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_dataset = TestDataset(root_dir=data_path,
                               resize_shape=[args.resize_shape, args.resize_shape],
                               image_processor=processor,
                               latent_res=args.latent_res,
                               n_classes=args.n_classes,
                               mask_res=args.mask_res,
                               use_data_aug=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

    return test_dataloader