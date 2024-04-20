import os
import torch
from model.tokenizer import load_tokenizer
from data.dataset_multi import TrainDataset_Seg, TestDataset_Seg
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def call_dataset(args) :

    # [1.2] image_processor
    tokenizer = load_tokenizer()
    clip_image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    if args.image_processor == 'clip':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif args.image_processor == 'vit':
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    elif args.image_processor == 'blip' :
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        processor = transforms.Compose([transforms.Resize((384,384), interpolation=InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        normalize, ])
    elif args.image_processor == 'resnet' :
        processor = transforms.Compose([transforms.Resize((256,256), interpolation=InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


    # [2] train & test dataset
    train_dataset = TrainDataset_Seg(root_dir=args.train_data_path,
                                         resize_shape=[args.resize_shape,args.resize_shape],
                                         tokenizer=tokenizer,
                                         image_processor=processor,
                                         latent_res=args.latent_res,
                                         n_classes = args.n_classes,
                                         mask_res = args.mask_res,
                                         use_data_aug = args.use_data_aug,)
    test_dataset = TestDataset_Seg(root_dir=args.test_data_path,
                                       resize_shape=[args.resize_shape,args.resize_shape],
                                       tokenizer=tokenizer,
                                       image_processor=processor,
                                       latent_res=args.latent_res,
                                       n_classes=args.n_classes,
                                       mask_res = args.mask_res,
                                       use_data_aug = False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False)
    return train_dataloader, test_dataloader, tokenizer