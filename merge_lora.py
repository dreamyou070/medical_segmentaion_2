""" Context-aware Network """
# 일부러 전략을 세워가면서 하자.
# 1.1. extracting edge feature
# 1.2. deep features are extracted according to edge feature
# 1.3 will i fine tuning with image condition ?
import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
import torch
from torch import nn
import os
from attention_store import AttentionStore
from data import call_dataset
from model import call_model_package
from model.segmentation_unet import SemanticModel
from model.diffusion_model import transform_models_if_DDP
from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads_3D_4D
from utils.attention_control import passing_argument, register_attention_control
from utils.accelerator_utils import prepare_accelerator
from utils.optimizer import get_optimizer, get_scheduler_fix
from utils.saving import save_model
from utils.loss import FocalLoss, Multiclass_FocalLoss
from utils.evaluate import evaluation_check
from monai.losses import DiceLoss, DiceCELoss
from model.focus_net import PFNet
from model.vision_condition_head import vision_condition_head
from model.positioning import AllPositioning
from model.pe import AllPositionalEmbedding
from transformers import ViTModel
from polyppvt.lib.pvt import PolypPVT
from model.lora import create_network
from safetensors.torch import load_file
from model.diffusion_model import load_target_model
# image conditioned segmentation mask generating

import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
import torch
import re
from model.lora import LoRANetwork
from model.lora_teacher import create_teacher_network

def main(args):

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(args.logging_dir, exist_ok=True)
    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f'\n step 2. preparing accelerator')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process


    print(f' (1) call student lora weight dict')
    base = r'/share0/dreamyou070/dreamyou070/MultiSegmentation/result/medical/leader_polyp'
    class_0_base = os.path.join(base, 'Pranet_Sub0')
    class_1_base = os.path.join(base, 'Pranet_Sub1')
    class_2_base = os.path.join(base, 'Pranet_Sub2')
    class_3_base = os.path.join(base, 'Pranet_Sub3')
    class_4_base = os.path.join(base, 'Pranet_Sub4')
    network_0_state_dict_dir = os.path.join(class_0_base, 'up_16_32_64_20240501/3_class_0_pvt_image_encoder/model/lora-000086.safetensors')
    network_1_state_dict_dir = os.path.join(class_1_base, 'up_16_32_64_20240501/3_class_3_pvt_image_encoder/model/lora-000001.safetensors')
    network_2_state_dict_dir = os.path.join(class_2_base, 'up_16_32_64_20240501/3_class_2_pvt_image_encoder/model/lora-000001.safetensors')
    network_3_state_dict_dir = os.path.join(class_3_base, 'up_16_32_64_20240501/3_class_3_pvt_image_encoder/model/lora-000001.safetensors')
    network_4_state_dict_dir = os.path.join(class_4_base, 'up_16_32_64_20240501/3_class_4_pvt_image_encoder/model/lora-000001.safetensors')
    network_weights = [network_0_state_dict_dir, network_1_state_dict_dir, network_2_state_dict_dir, network_3_state_dict_dir, network_4_state_dict_dir]

    print(f' (2) make student networks')
    weight_dtype, save_dtype = prepare_dtype(args)
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value
    if args.image_processor == 'vit': # ViTModel
        image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    elif args.image_processor == 'pvt':
        model = PolypPVT()
        pretrained_pth_path = '/share0/dreamyou070/dreamyou070/PolypPVT/Polyp_PVT/model_pth/PolypPVT.pth'
        model.load_state_dict(torch.load(pretrained_pth_path))
        image_model = model.backbone  # pvtv2_b2 model
    image_model.requires_grad_(False)
    condition_model = image_model # image model is a condition
    condition_modality = 'image'

    student_nets = []
    for net_weight_dir in network_weights:
        student_net = create_network(1.0,
                                     args.network_dim,
                                     args.network_alpha,
                                     vae,
                                     condition_model=condition_model,
                                     unet=unet,
                                     neuron_dropout=args.network_dropout,
                                     condition_modality=condition_modality,)
        student_net.load_weights(net_weight_dir)
        student_net.requires_grad_(False)
        student_nets.append(student_net)

    print(f' (3) make teacher networks')
    teacher_network = create_teacher_network(multiplier = 1.0,
                                             network_dim = args.network_dim,
                                             network_alpha =args.network_alpha,
                                             vae = vae,
                                             condition_model = condition_model,
                                             unet=unet,
                                             neuron_dropout=args.network_dropout,
                                             condition_modality=condition_modality,
                                             student_networks=student_nets,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    # step 2. dataset
    parser.add_argument("--resize_shape", type=int, default=512)
    parser.add_argument('--train_data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--test_data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument("--latent_res", type=int, default=64)
    # step 3. preparing accelerator
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--lowram", action="store_true", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--pe_do_concat", action='store_true')
    parser.add_argument("--d_dim", default=320, type=int)
    # step 4. model
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='facebook/diffusion-dalle')
    parser.add_argument("--clip_skip", type=int, default=None,
                        help="use output of nth layer from back of text encoder (n>=1)")
    parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                        help="max token length of text encoder (default for 75, 150 or 225) / text encoder", )
    parser.add_argument("--vae_scale_factor", type=float, default=0.18215)
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_dim", type=int, default=64, help="network dimensions (depends on each network) ")
    parser.add_argument("--network_alpha", type=float, default=4,
                        help="alpha for LoRA weight scaling, default 1 ", )
    parser.add_argument("--network_dropout", type=float, default=None, )
    parser.add_argument("--network_args", type=str, default=None, nargs="*", )
    parser.add_argument("--dim_from_weights", action="store_true", )
    parser.add_argument("--n_classes", default=4, type=int)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--mask_res", type=int, default=128)
    # step 5. optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                        help="AdamW , AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov,"
                             "SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP,"
                             "DAdaptLion, DAdaptSGD, AdaFactor", )
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="use 8bit AdamW optimizer(requires bitsandbytes)", )
    parser.add_argument("--use_lion_optimizer", action="store_true",
                        help="use Lion optimizer (requires lion-pytorch)", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping")
    parser.add_argument("--optimizer_args", type=str, default=None, nargs="*",
                        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") ', )
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module")
    parser.add_argument("--lr_scheduler_args", type=str, default=None, nargs="*",
                        help='additional arguments for scheduler (like "T_max=100")')
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", help="scheduler to use for lr")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler (default is 0)", )
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                        help="Number of restarts for cosine scheduler with restarts / cosine with restarts", )
    parser.add_argument("--lr_scheduler_power", type=float, default=1,
                        help="Polynomial power for polynomial scheduler / polynomial", )
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--unet_lr', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--train_unet', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    # step 10. training
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors)", )
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--position_embedder_weights", type=str, default=None)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--check_training", action='store_true')
    parser.add_argument("--pretrained_segmentation_model", type=str)
    parser.add_argument("--use_batchnorm", action='store_true')
    parser.add_argument("--use_instance_norm", action='store_true')
    parser.add_argument("--use_layer_norm", action='store_true')
    parser.add_argument("--aggregation_model_c", action='store_true')
    parser.add_argument("--aggregation_model_d", action='store_true')
    parser.add_argument("--norm_type", type=str, default='batchnorm',
                        choices=['batch_norm', 'instance_norm', 'layer_norm'])
    parser.add_argument("--non_linearity", type=str, default='relu', choices=['relu', 'leakyrelu', 'gelu'])
    parser.add_argument("--neighbor_size", type=int, default=3)
    parser.add_argument("--do_semantic_position", action='store_true')
    parser.add_argument("--use_init_query", action='store_true')
    parser.add_argument("--use_dice_loss", action='store_true')
    parser.add_argument("--use_patch", action='store_true')
    parser.add_argument("--use_monai_focal_loss", action='store_true')
    parser.add_argument("--use_data_aug", action='store_true')
    parser.add_argument("--deactivating_loss", action='store_true')
    parser.add_argument("--use_dice_ce_loss", action='store_true')
    parser.add_argument("--dice_weight", type=float, default=1)
    parser.add_argument("--segmentation_efficient", action='store_true')
    parser.add_argument("--binary_test", action='store_true')
    parser.add_argument("--attn_factor", type=int, default=3)
    parser.add_argument("--max_timestep", type=int, default=200)
    parser.add_argument("--min_timestep", type=int, default=0)
    parser.add_argument("--use_noise_regularization", action='store_true')
    parser.add_argument("--use_cls_token", action='store_true')
    parser.add_argument("--independent_decoder", action='store_true')
    parser.add_argument("--high_latent_feature", action='store_true')
    parser.add_argument("--use_patch_discriminator", action='store_true')
    parser.add_argument("--init_latent_p", type=float, default=1)
    parser.add_argument("--generator_loss_weight", type=float, default=1)
    parser.add_argument("--segmentation_loss_weight", type=float, default=1)
    parser.add_argument("--use_image_by_caption", action='store_true')
    parser.add_argument("--gt_ext_npy", action='store_true')
    parser.add_argument("--generation", action='store_true')
    parser.add_argument("--test_like_train", action='store_true')
    parser.add_argument("--test_before_query", action='store_true')
    parser.add_argument("--do_text_attn", action='store_true')
    parser.add_argument("--use_image_condition", action='store_true')
    parser.add_argument("--use_text_condition", action='store_true')
    parser.add_argument("--image_processor", default='vit', type=str)
    parser.add_argument("--image_model_training", action='store_true')
    parser.add_argument("--erase_position_embeddings", action='store_true')
    parser.add_argument("--light_decoder", action='store_true')
    parser.add_argument("--use_base_prompt", action='store_true')
    parser.add_argument("--use_noise_pred_loss", action='store_true')
    parser.add_argument("--use_vit_pix_embed", action='store_true')
    parser.add_argument("--not_use_cls_token", action='store_true')
    parser.add_argument("--without_condition", action='store_true')
    parser.add_argument("--only_use_cls_token", action='store_true')
    parser.add_argument("--reducing_redundancy", action='store_true')
    parser.add_argument("--use_weighted_reduct", action='store_true')
    parser.add_argument("--reverse", action='store_true')
    parser.add_argument("--online_pseudo_loss", action='store_true')
    parser.add_argument("--only_online_pseudo_loss", action='store_true')
    parser.add_argument("--pseudo_loss_weight", type=float, default=1)
    parser.add_argument("--anomal_loss_weight", type=float, default=1)
    parser.add_argument("--anomal_mse_loss", action='store_true')
    parser.add_argument("--use_self_attn", action='store_true')
    parser.add_argument("--use_positioning_module", action='store_true')
    parser.add_argument("--use_channel_attn", action='store_true')
    parser.add_argument("--use_simple_segmodel", action='store_true')
    parser.add_argument("--use_segmentation_model", action='store_true')
    parser.add_argument("--use_max_for_focus_map", action='store_true')
    parser.add_argument("--positioning_module_weights", type=str, default=None)
    parser.add_argument("--vision_head_weights", type=str, default=None)
    parser.add_argument("--segmentation_model_weights", type=str, default=None)
    parser.add_argument("--previous_positioning_module", action='store_true')
    parser.add_argument("--save_image", action='store_true')
    parser.add_argument("--use_one", action='store_true')
    parser.add_argument("--channel_spatial_cascaded", action='store_true')
    parser.add_argument("--base_path", type=str)
    args = parser.parse_args()
    passing_argument(args)
    from data.dataset import passing_mvtec_argument

    passing_mvtec_argument(args)
    main(args)