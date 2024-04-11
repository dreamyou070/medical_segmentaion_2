import os
import torch
from data.dataset_multi import TrainDataset_Seg
from model.tokenizer import load_tokenizer

def call_dataset(args) :

    # [1] load tokenizer
    tokenizer = load_tokenizer(args)

    # [2] train & test dataset
    train_dataset = TrainDataset_Seg(root_dir=args.train_data_path,
                                     resize_shape=[args.resize_shape,args.resize_shape],
                                     tokenizer=tokenizer,
                                     caption=args.trigger_word,
                                     latent_res=args.latent_res,
                                     n_classes = args.n_classes,
                                     mask_res = args.mask_res,)
    test_dataset = TrainDataset_Seg(root_dir=args.test_data_path,
                                    resize_shape=[args.resize_shape,args.resize_shape],
                                    tokenizer=tokenizer,
                                    caption=args.trigger_word,
                                    latent_res=args.latent_res,
                                    n_classes=args.n_classes,
                                    mask_res = args.mask_res,)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False)
    return train_dataloader, test_dataloader