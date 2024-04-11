import os
from diffusers import StableDiffusionPipeline, AutoencoderKL
from model.unet import UNet2DConditionModel
from model.diffusion_model_conversion import (load_checkpoint_with_text_encoder_conversion,
                    convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint)
from model.diffusion_model_config import (create_unet_diffusers_config,create_vae_diffusers_config)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig, logging
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import gc

def load_models_from_stable_diffusion_checkpoint(ckpt_path, device="cpu",unet_use_linear_projection_in_v2=True):

    # [0]
    _, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path)

    # [1] unet
    unet_config = create_unet_diffusers_config(unet_use_linear_projection_in_v2)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, unet_config)
    unet = UNet2DConditionModel(**unet_config).to(device)
    info = unet.load_state_dict(converted_unet_checkpoint)
    print("loading u-net:", info)
    # [2] vae
    vae_config = create_vae_diffusers_config()
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
    vae = AutoencoderKL(**vae_config).to(device)
    info = vae.load_state_dict(converted_vae_checkpoint)
    print("loading vae:", info)

    # [3] text_encoder
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint(state_dict)
    cfg = CLIPTextConfig(vocab_size=49408,hidden_size=768,intermediate_size=3072,num_hidden_layers=12,
                         num_attention_heads=12,max_position_embeddings=77,
                         hidden_act="quick_gelu",layer_norm_eps=1e-05,dropout=0.0,
                         attention_dropout=0.0,initializer_range=0.02,initializer_factor=1.0,
                         pad_token_id=1,bos_token_id=0,eos_token_id=2,
                         model_type="clip_text_model",projection_dim=768,torch_dtype="float32",)
    text_model = CLIPTextModel._from_config(cfg)
    converted_text_encoder_checkpoint.pop('text_model.embeddings.position_ids')
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    print("loading text encoder:", info)
    return text_model, vae, unet

def _load_target_model(args: argparse.Namespace, weight_dtype,
                       device="cpu", unet_use_linear_projection_in_v2=False):
    name_or_path = args.pretrained_model_name_or_path
    name_or_path = os.path.realpath(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers
    print(f"load StableDiffusion checkpoint: {name_or_path}")
    text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(name_or_path, device,
                                                     unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2)
    return text_encoder, vae, unet, load_stable_diffusion_format

def transform_if_model_is_DDP(text_encoder, unet, network=None):
    return (model.module if type(model) == DDP else model for model in [text_encoder, unet, network] if model is not None)
def load_target_model(args, weight_dtype, accelerator, unet_use_linear_projection_in_v2=False):
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            print(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")
            text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(
                args,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2, )
            gc.collect()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    text_encoder, unet = transform_if_model_is_DDP(text_encoder, unet)
    return text_encoder, vae, unet, load_stable_diffusion_format


def transform_models_if_DDP(models):
    from torch.nn.parallel import DistributedDataParallel as DDP
    return [model.module if type(model) == DDP else model for model in models if model is not None]