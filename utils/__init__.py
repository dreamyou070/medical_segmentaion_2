import torch
import os
import ast
import argparse

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


def reshape_batch_dim_to_heads(tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = 8
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)  # 1,8,pix_num, dim -> 1,pix_nun, 8,dim
    tensor = tensor.permute(0, 2, 1, 3).contiguous().reshape(batch_size // head_size, seq_len,
                                                             dim * head_size)  # 1, pix_num, long_dim
    res = int(seq_len ** 0.5)
    tensor = tensor.view(batch_size // head_size, res, res, dim * head_size).contiguous()
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # 1, dim, res,res
    return tensor