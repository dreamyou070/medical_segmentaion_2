import torch
import os
import ast
import argparse
import numpy as np
from PIL import Image
def arg_as_list(arg):
    v = ast.literal_eval(arg)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
    return v

def prepare_dtype(args):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32
    return weight_dtype, save_dtype


def reshape_batch_dim_to_heads_3D_4D(tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = 8
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)  # 1,8,pix_num, dim -> 1,pix_nun, 8,dim
    tensor = tensor.permute(0, 2, 1, 3).contiguous().reshape(batch_size // head_size, seq_len,
                                                             dim * head_size)  # 1, pix_num, long_dim
    res = int(seq_len ** 0.5)
    tensor = tensor.view(batch_size // head_size, res, res, dim * head_size).contiguous()
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # 1, dim, res,res
    return tensor

def reshape_batch_dim_to_heads_3D_3D(tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = 8
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)  # 1,8,pix_num, dim -> 1,pix_nun, 8,dim
    tensor = tensor.permute(0, 2, 1, 3).contiguous().reshape(batch_size // head_size,
                                                             seq_len,
                                                             dim * head_size)  # 1, pix_num, long_dim
    return tensor


def get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, noise = None):
    # Sample noise that we'll add to the latents
    if noise is None:
        noise = torch.randn_like(latents, device=latents.device)


    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep

    timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=latents.device)
    timesteps = timesteps.long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return noise, noisy_latents, timesteps

def torch_to_pil(torch_img, binary):
    # torch_img = [3, H, W], from -1 to 1
    if torch_img.dim() == 3:
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
    else:
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8)

    if binary :
        np_img = np.where(np_img > 127, 255, 0)
        pil = Image.fromarray(np_img).convert("L").convert('RGB')
    else :
        pil = Image.fromarray(np_img).convert("RGB")
    return pil


