import os
import torch
from model.tokenizer import load_tokenizer
from data.dataset_multi import TrainDataset_Seg, TestDataset_Seg
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel



#image_path = 'data_sample/image/sample_200.jpg'
#image = Image.open(image_path)
#inputs = processor(images=image, return_tensors="pt", padding=True)#.data['pixel_values'] # [1,3,224,224]
#image_pixel = inputs.data['pixel_values']
#print(f'image_pixel = {image_pixel.shape}')
#clip_image_model.to("cuda:0")
#image_inputs = inputs.to("cuda:0")

#image_features = clip_image_model.get_image_features(**inputs) # [batch, pix_num, dim = 784]
#print(f'image_features = {image_features.shape}')


def call_dataset(args) :

    # [1.1] load tokenizer
    tokenizer = load_tokenizer(args)
    # [1.2] image_processor
    #clip_image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # [2] train & test dataset
    train_dataset = TrainDataset_Seg(root_dir=args.train_data_path,
                                     resize_shape=[args.resize_shape,args.resize_shape],
                                     tokenizer=tokenizer,
                                     imagee_processor=processor,
                                     latent_res=args.latent_res,
                                     n_classes = args.n_classes,
                                     mask_res = args.mask_res,
                                     use_data_aug = args.use_data_aug,)
    test_dataset = TestDataset_Seg(root_dir=args.test_data_path,
                                    resize_shape=[args.resize_shape,args.resize_shape],
                                    tokenizer=tokenizer,
                                   imagee_processor=processor,
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