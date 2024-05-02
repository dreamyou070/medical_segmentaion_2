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

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")

BLOCKS = ["text_model",
          "unet_down_blocks_0_attentions_0","unet_down_blocks_0_attentions_1",
          "unet_down_blocks_1_attentions_0","unet_down_blocks_1_attentions_1",
          "unet_down_blocks_2_attentions_0","unet_down_blocks_2_attentions_1",
          "unet_mid_block_attentions_0",
          "unet_up_blocks_1_attentions_0","unet_up_blocks_1_attentions_1","unet_up_blocks_1_attentions_2",
          "unet_up_blocks_2_attentions_0","unet_up_blocks_2_attentions_1","unet_up_blocks_2_attentions_2",
          "unet_up_blocks_3_attentions_0","unet_up_blocks_3_attentions_1","unet_up_blocks_3_attentions_2", ]

def gcd(a, b):
    for i in range(min(a, b), 0, -1):
        if a % i == 0 and b % i == 0:
            return i


class TeacherLoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self,
                 lora_name,
                 org_module: torch.nn.Module,
                 multiplier=1.0,
                 lora_dim=4,
                 alpha=1,
                 dropout=None,
                 rank_dropout=None,module_dropout=None,
                 student_modules=None):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.is_linear = False
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.is_linear = True
        self.in_dim = in_dim
        self.out_dim = out_dim

        common_dim = gcd(in_dim, out_dim)
        self.common_dim = common_dim
        down_dim = int(in_dim // common_dim)
        up_dim = int(out_dim // common_dim)

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False) # necessary value
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)  #

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.org_weight = org_module.weight.detach().clone() #####################################################
        self.org_module_ref = [org_module]  ########################################################################

        self.alpha_1 = nn.Parameter(torch.tensor(1.0))
        self.alpha_2 = nn.Parameter(torch.tensor(1.0))
        self.alpha_3 = nn.Parameter(torch.tensor(1.0))
        self.alpha_4 = nn.Parameter(torch.tensor(1.0))
        self.alpha_5 = nn.Parameter(torch.tensor(1.0))
        
        self.alphas = [self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4, self.alpha_5]
        # self.betas = [nn.Parameter(torch.tensor(1.0)) for _ in range(len(student_modules))]

        for i, student_module in enumerate(student_modules):
            student_module.requires_grad = False
        self.student_modules = student_modules

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        #del self.org_module

    def restore(self):
        self.org_module.forward = self.org_forward

    def forward(self, x):


        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded
        value = 0
        for alpha, module in zip(self.alphas, self.student_modules) :
            lora_value = module.lora_up.to(x.device)(module.lora_down.to(x.device)(x))
            value += alpha.to(x.device) * lora_value

        lx = value

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
            # rank dropout

        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask
            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        """
        # normal dropout
        

        for beta, module in zip(self.betas, self.student_modules) :
            lx += beta.to(x.device) * module.lora_up.to(x.device)(lx)
        """

        return org_forwarded + lx * self.multiplier * scale


class LoRAInfModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self,lora_name,
                 org_module: torch.nn.Module,
                 multiplier=1.0,
                 lora_dim=4,
                 alpha=1,
                 dropout=None,
                 rank_dropout=None,module_dropout=None,):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.is_linear = False
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.is_linear = True
        self.in_dim = in_dim
        self.out_dim = out_dim

        common_dim = gcd(in_dim, out_dim)
        self.common_dim = common_dim
        down_dim = int(in_dim // common_dim)
        up_dim = int(out_dim // common_dim)

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.org_weight = org_module.weight.detach().clone() #####################################################
        self.org_module_ref = [org_module]  ########################################################################

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        #del self.org_module

    def restore(self):
        self.org_module.forward = self.org_forward

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale


def parse_block_lr_kwargs(nw_kwargs):
    down_lr_weight = nw_kwargs.get("down_lr_weight", None)
    mid_lr_weight = nw_kwargs.get("mid_lr_weight", None)
    up_lr_weight = nw_kwargs.get("up_lr_weight", None)

    # 以上のいずれにも設定がない場合は無効としてNoneを返す
    if down_lr_weight is None and mid_lr_weight is None and up_lr_weight is None:
        return None, None, None

    # extract learning rate weight for each block
    if down_lr_weight is not None:
        # if some parameters are not set, use zero
        if "," in down_lr_weight:
            down_lr_weight = [(float(s) if s else 0.0) for s in down_lr_weight.split(",")]

    if mid_lr_weight is not None:
        mid_lr_weight = float(mid_lr_weight)

    if up_lr_weight is not None:
        if "," in up_lr_weight:
            up_lr_weight = [(float(s) if s else 0.0) for s in up_lr_weight.split(",")]

    down_lr_weight, mid_lr_weight, up_lr_weight = get_block_lr_weight(
        down_lr_weight, mid_lr_weight, up_lr_weight, float(nw_kwargs.get("block_lr_zero_threshold", 0.0))
    )

    return down_lr_weight, mid_lr_weight, up_lr_weight





# 外部から呼び出す可能性を考慮しておく

def get_block_index(lora_name: str) -> int:
    block_idx = -1  # invalid lora name

    m = RE_UPDOWN.search(lora_name)
    if m:
        g = m.groups()
        i = int(g[1])
        j = int(g[3])
        if g[2] == "resnets":
            idx = 3 * i + j
        elif g[2] == "attentions":
            idx = 3 * i + j
        elif g[2] == "upsamplers" or g[2] == "downsamplers":
            idx = 3 * i + 2

        if g[0] == "down":
            block_idx = 1 + idx  # 0に該当するLoRAは存在しない
        elif g[0] == "up":
            block_idx = LoRANetwork.NUM_OF_BLOCKS + 1 + idx

    elif "mid_block_" in lora_name:
        block_idx = LoRANetwork.NUM_OF_BLOCKS  # idx=12

    return block_idx

def create_teacher_network(multiplier: float,
                   network_dim: Optional[int],
                   network_alpha: Optional[float],
                   vae: AutoencoderKL,
                   condition_model: Union[CLIPTextModel, List[CLIPTextModel]],
                   unet,
                   neuron_dropout: Optional[float] = None,
                   condition_modality = 'text',
                   student_networks = None,
                   **kwargs,):

    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # block dim/alpha/lr
    block_dims = kwargs.get("block_dims", None)
    down_lr_weight, mid_lr_weight, up_lr_weight = parse_block_lr_kwargs(kwargs)

    # 以上のいずれかに指定があればblockごとのdim(rank)を有効にする
    if block_dims is not None or down_lr_weight is not None or mid_lr_weight is not None or up_lr_weight is not None:
        block_alphas = kwargs.get("block_alphas", None)
        conv_block_dims = kwargs.get("conv_block_dims", None)
        conv_block_alphas = kwargs.get("conv_block_alphas", None)

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = get_block_dims_and_alphas(
            block_dims, block_alphas, network_dim, network_alpha, conv_block_dims, conv_block_alphas, conv_dim, conv_alpha
        )

        # remove block dim/alpha without learning rate
        block_dims, block_alphas, conv_block_dims, conv_block_alphas = remove_block_dims_and_alphas(
            block_dims, block_alphas, conv_block_dims, conv_block_alphas, down_lr_weight, mid_lr_weight, up_lr_weight
        )

    else:
        block_alphas = None
        conv_block_dims = None
        conv_block_alphas = None

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    net_key_names = kwargs.get('key_layers', None)
    # すごく引数が多いな ( ^ω^)･･･
    network = TeacherLoRANetwork(condition_model=condition_model,
                          unet=unet,
                          multiplier=multiplier,
                          lora_dim=network_dim,
                          alpha=network_alpha,
                          dropout=neuron_dropout,
                          rank_dropout=rank_dropout,
                          module_dropout=module_dropout,
                          conv_lora_dim=conv_dim,
                          conv_alpha=conv_alpha,
                          block_dims=block_dims,
                          block_alphas=block_alphas,
                          conv_block_dims=conv_block_dims,
                          conv_block_alphas=conv_block_alphas,
                          varbose=True,
                          net_key_names=net_key_names,
                          condition_modality=condition_modality,
                          student_networks=student_networks)

    if up_lr_weight is not None or mid_lr_weight is not None or down_lr_weight is not None:
        network.set_block_lr_weight(up_lr_weight, mid_lr_weight, down_lr_weight)

    return network

class TeacherLoRANetwork(torch.nn.Module):
    #
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    UNET_TEXT_PART = 'attentions_0'

    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    IMAGE_ENCODER_TARGET_REPLACE_MODULE = ["ViTSelfAttention",
                                           "ViTPooler",
                                           "ViTSelfOutput",
                                           "ViTIntermediate",
                                           "Attention",
                                           "Mlp"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    LORA_PREFIX_IMAGE_ENCODER = "lora_im"

    def __init__(
        self,
        condition_model: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        block_wise : Optional[List[int]] = None,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        block_dims: Optional[List[int]] = None,
        block_alphas: Optional[List[float]] = None,
        conv_block_dims: Optional[List[int]] = None,
        conv_block_alphas: Optional[List[float]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        # LoRAInfModule
        module_class: Type[object] = TeacherLoRAModule,
        varbose: Optional[bool] = False,
        net_key_names: Optional[bool] = False,
        condition_modality='text',
        student_networks=None) -> None:

        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        if modules_dim is not None:
            print(f"create LoRA network from weights")
        elif block_dims is not None:
            print(f"create LoRA network from block_dims")
            print(f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            print(f"block_dims: {block_dims}")
            print(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                print(f"conv_block_dims: {conv_block_dims}")
                print(f"conv_block_alphas: {conv_block_alphas}")
        else:
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            print(f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            if self.conv_lora_dim is not None:
                print(f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")

        # create module instances
        def create_modules(is_unet: bool,
                           text_encoder_idx: Optional[int],  # None, 1, 2
                           root_module: torch.nn.Module,
                           target_replace_modules : List[torch.nn.Module],
                           prefix,
                           student_networks : List[TeacherLoRAModule] ) :

            loras = []
            skipped = []

            for name, module in root_module.named_modules():

                if module.__class__.__name__ in target_replace_modules:

                    for child_name, child_module in module.named_modules():

                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:

                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")


                            # --------------------------------------------------------------------------------------------------
                            # [2] set dim and alpha
                            dim = None
                            alpha = None
                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                block_idx = get_block_index(lora_name) # block
                                if is_linear or is_conv2d_1x1:
                                    dim = block_dims[block_idx]
                                    alpha = block_alphas[block_idx]
                                elif conv_block_dims is not None:
                                    dim = conv_block_dims[block_idx]
                                    alpha = conv_block_alphas[block_idx]
                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha
                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(lora_name)
                                continue

                            # --------------------------------------------------------------------------------------------------
                            # [3] make lora module
                            # im module = 80
                            student_modules = []

                            for student_lora in student_networks:
                                for student_lora in (student_lora.image_encoder_loras + student_lora.unet_loras):
                                    if student_lora.lora_name == lora_name:
                                        student_lora.lora_name = None
                                        student_modules.append(student_lora)
                            if block_wise == None :
                                lora = module_class(lora_name,
                                                    child_module,
                                                    self.multiplier,
                                                    dim,
                                                    alpha,
                                                    dropout=dropout,
                                                    rank_dropout=rank_dropout,
                                                    module_dropout=module_dropout,
                                                    student_modules=student_modules)
                                loras.append(lora)
                            else :
                                for i, block in enumerate(BLOCKS) :
                                    if block in lora_name and block_wise[i] == 1:
                                        lora = module_class(lora_name,
                                                            child_module,
                                                            self.multiplier,
                                                            dim,
                                                            alpha,
                                                            dropout=dropout,
                                                            rank_dropout=rank_dropout,
                                                            module_dropout=module_dropout,
                                                            student_modules=student_modules)
                                        loras.append(lora)
            return loras, skipped

        # ------------------------------------------------------------------------------------------------------------------------
        # [1] Unet
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
        self.unet_loras, skipped_un = create_modules(True,
                                                     None,
                                                     unet,
                                                     target_replace_modules=target_modules,
                                                     prefix=LoRANetwork.LORA_PREFIX_UNET,
                                                     student_networks=student_networks)
        # ------------------------------------------------------------------------------------------------------------------------
        # [1] text encoder
        if condition_modality == 'text':
            text_encoder = condition_model
            text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]
            self.text_encoder_loras = []
            skipped_te = []  # 1 model
            for i, text_encoder in enumerate(text_encoders):
                if len(text_encoders) > 1:
                    index = i + 1
                    print(f"create LoRA for Text Encoder {index}:")
                else:
                    index = None
                    # print(f"create LoRA for Text Encoder:") # Here is the problem
                if condition_modality == 'image':
                    prefix_ = LoRANetwork.LORA_PREFIX_IMAGE_ENCODER
                    target_replace_module_condition = LoRANetwork.IMAGE_ENCODER_TARGET_REPLACE_MODULE
                else:
                    prefix_ = LoRANetwork.LORA_PREFIX_TEXT_ENCODER
                    target_replace_module_condition = LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
                text_encoder_loras, skipped = create_modules(False,
                                                             index,
                                                             text_encoder,
                                                             target_replace_modules=target_replace_module_condition,
                                                             prefix=prefix_,
                                                             student_networks=student_networks)
                self.text_encoder_loras.extend(text_encoder_loras)
                skipped_te += skipped
            print(f"create LoRA for Text Encoder : {len(self.text_encoder_loras)} modules.")  # Here (61 modules)
            skipped = skipped_te + skipped_un

            # assertion
            names = set()
            for lora in self.text_encoder_loras + self.unet_loras:
                assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
                names.add(lora.lora_name)

        else :
            image_condition = condition_model
            image_encoders = image_condition if type(image_condition) == list else [image_condition]
            self.image_encoder_loras = []
            skipped_ie = []  # 1 model
            if image_condition is not None:
                for i, image_encoder in enumerate(image_encoders):
                    if len(image_encoders) > 1:
                        index = i + 1
                        print(f"create LoRA for Image Encoder {index}:")
                    else:
                        index = None
                    # ---------------------------------------------------------------------------------------------------------------------
                    # create image encoder LoRA
                    prefix_ = LoRANetwork.LORA_PREFIX_IMAGE_ENCODER
                    target_replace_module_condition = LoRANetwork.IMAGE_ENCODER_TARGET_REPLACE_MODULE

                    image_encoder_loras, skipped = create_modules(False,
                                                                  index,
                                                                  root_module=image_encoder,
                                                                  target_replace_modules=target_replace_module_condition,
                                                                  prefix=prefix_,
                                                                  student_networks=student_networks)
                    self.image_encoder_loras.extend(image_encoder_loras)
                    skipped_ie += skipped
                print(f"create LoRA for Image Encoder : {len(self.image_encoder_loras)} modules.")
                skipped = skipped_ie + skipped_un

                # assertion
                names = set()
                total_loras = self.image_encoder_loras + self.unet_loras
                print(f'len of total_loras : {len(total_loras)}')
                for lora in total_loras :
                    assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
                    names.add(lora.lora_name)

        # ------------------------------------------------------------------------------------------------------------------------
        if varbose and len(skipped) > 0:
            print(f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:")
            for name in skipped:
                print(f"\t{name}")

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False


    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        info = self.load_state_dict(weights_sd, False)
        return info

    def restore(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            lora.restore()


    def apply_to(self, text_encoder, unet, apply_condition_model=True, apply_unet=True, condition_modality = 'text'):



        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        if condition_modality == 'text':
            if apply_condition_model:
                print("enable LoRA for text encoder")
            else:
                self.text_encoder_loras = []
            for lora in self.text_encoder_loras + self.unet_loras:
                lora.apply_to()
                self.add_module(lora.lora_name, lora)

        elif condition_modality == 'image':
            if apply_condition_model:
                print("enable LoRA for image encoder")
            else:
                self.image_encoder_loras = []

            for lora in self.image_encoder_loras + self.unet_loras:
                lora.apply_to()
                self.add_module(lora.lora_name, lora)

    def restore(self, condition_modality = 'text'):

        if condition_modality == 'text':

            for lora in self.text_encoder_loras + self.unet_loras:
                lora.restore()

        elif condition_modality == 'image':

            for lora in self.image_encoder_loras + self.unet_loras:
                lora.restore()

    def is_mergeable(self):
        return True

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        print(f"weights are merged")

    # 層別学習率用に層ごとの学習率に対する倍率を定義する　引数の順番が逆だがとりあえず気にしない
    def set_block_lr_weight(
        self,
        up_lr_weight: List[float] = None,
        mid_lr_weight: float = None,
        down_lr_weight: List[float] = None,
    ):
        self.block_lr = True
        self.down_lr_weight = down_lr_weight
        self.mid_lr_weight = mid_lr_weight
        self.up_lr_weight = up_lr_weight

    def get_lr_weight(self, lora: TeacherLoRAModule) -> float:
        lr_weight = 1.0
        block_idx = get_block_index(lora.lora_name)
        if block_idx < 0:
            return lr_weight

        if block_idx < LoRANetwork.NUM_OF_BLOCKS: # 12
            if self.down_lr_weight != None:
                lr_weight = self.down_lr_weight[block_idx]
        elif block_idx == LoRANetwork.NUM_OF_BLOCKS:
            if self.mid_lr_weight != None:
                lr_weight = self.mid_lr_weight
        elif block_idx > LoRANetwork.NUM_OF_BLOCKS:
            if self.up_lr_weight != None:
                lr_weight = self.up_lr_weight[block_idx - LoRANetwork.NUM_OF_BLOCKS - 1]

        return lr_weight

    # 二つのText Encoderに別々の学習率を設定できるようにするといいかも

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr,
                                 condition_modality = 'text'):

        self.requires_grad_(True)
        all_params = [] # list

        def enumerate_params(loras):
            params = []
            for lora in loras:
                for name, param in lora.named_parameters():
                    print(f'parameter name = {name}')
                    #if 'alpha' in name :
                        #params.extend(param)
            return params

        if condition_modality == 'text':

            if self.text_encoder_loras:
                param_data = {"params": enumerate_params(self.text_encoder_loras)}
                if text_encoder_lr is not None:
                    param_data["lr"] = text_encoder_lr
                all_params.append(param_data) # len 2 (unet, image_encoder)

        elif condition_modality == 'image':
            if self.image_encoder_loras:
                param_data = {"params": enumerate_params(self.image_encoder_loras)}
                if text_encoder_lr is not None:
                    param_data["lr"] = text_encoder_lr
                all_params.append(param_data)

        if self.unet_loras:

            if self.block_lr:
                block_idx_to_lora = {}
                for lora in self.unet_loras:
                    idx = get_block_index(lora.lora_name)
                    if idx not in block_idx_to_lora:
                        block_idx_to_lora[idx] = []
                    block_idx_to_lora[idx].append(lora)
                for idx, block_loras in block_idx_to_lora.items():
                    param_data = {"params": enumerate_params(block_loras)}
                    if unet_lr is not None:
                        final_lr = unet_lr * self.get_lr_weight(block_loras[0])
                        param_data["lr"] = final_lr
                    elif default_lr is not None:
                        param_data["lr"] = default_lr * self.get_lr_weight(block_loras[0])
                    if ("lr" in param_data) and (param_data["lr"] == 0):
                        continue
                    all_params.append(param_data)

            else:
                param_data = {"params": enumerate_params(self.unet_loras)}
                if unet_lr is not None:
                    param_data["lr"] = unet_lr
                all_params.append(param_data)

        return all_params


    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            if metadata is None:
                metadata = {}
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        # 重みのバックアップを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    """
    def restore_weights(self):
        # 重みのリストアを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            lora.with_lora = False
            org_module = lora.org_module_ref[0]
            #if not org_module._lora_restored:
            sd = org_module.state_dict()
            #sd["weight"] = org_module._lora_org_weight
            sd["weight"] = lora.org_weight
            org_module.load_state_dict(sd)
            #
            #org_module._lora_restored = True
    """
    def restore_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            lora.restore_weight()

    def pre_calculation(self):
        # 事前計算を行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)