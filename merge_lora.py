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


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self,lora_name,
                 org_module: torch.nn.Module,
                 multiplier=1.0,
                 lora_dim=4,
                 alpha=1,
                 dropout=None,
                 rank_dropout=None,
                 module_dropout=None,
                 student_loras=None,):
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

        self.alphas = [nn.Parameters(1) for student_lora in student_loras] # alpha for lora_up
        self.betas = [nn.Parameters(1) for student_lora in student_loras]  # beta for lora_down

        self.student_loras = student_loras


    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        #del self.org_module

    def restore(self):
        self.org_module.forward = self.org_forward

    def forward(self, x):

        org_forwarded = self.org_forward(x)
        lx = 0
        for alpha, student_lora in zip(self.alphas, self.student_loras) :
            lx += alpha * student_lora.lora_down(x)
        for beta, student_lora in zip(self.betas, self.student_loras) :
            lx += beta * student_lora.lora_up(x)
        lx = self.lora_up(lx)
        scale = self.scale

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
"""
class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # 後から参照できるように
        self.enabled = True

        # check regional or not by lora_name

        # --------------------------------------------------------------------------------------------
        # text encoder lora
        self.text_encoder = False
        if lora_name.startswith("lora_te_"):
            self.regional = False
            self.use_sub_prompt = True
            self.text_encoder = True

        elif "attn2_to_k" in lora_name or "attn2_to_v" in lora_name:
            self.regional = False
            self.use_sub_prompt = True
        elif "time_emb" in lora_name:
            self.regional = False
            self.use_sub_prompt = False
        else:
            self.regional = True
            self.use_sub_prompt = False

        self.network: LoRANetwork = None
        self.org_weight = org_module.weight.detach().clone()

    def set_network(self, network):
        self.network = network

    # freezeしてマージする
    def merge_to(self, sd, dtype, device):
        # get up/down weight
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)
        down_weight = sd["lora_down.weight"].to(torch.float).to(device)

        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(torch.float)

        # merge weight
        if len(weight.size()) == 2:
            # linear
            weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                weight
                + self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale)
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # print(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + self.multiplier * conved * self.scale

        # set weight to org_module
        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    # 復元できるマージのため、このモジュールのweightを返す
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight

    def set_region(self, region):
        self.region = region
        self.region_mask = None

    def default_forward(self, x):
        # print("default_forward", self.lora_name, x.size())
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

    def forward(self, x):

        # --------------------------------------------------------------------------------------------------------
        # if not enable, just return original forward
        if not self.enabled:
            return self.org_forward(x)


        if self.network is None or self.network.sub_prompt_index is None:
            return self.default_forward(x)

        if not self.regional and not self.use_sub_prompt:
            return self.default_forward(x)

        if self.regional:
            return self.regional_forward(x)
        else:
            return self.sub_prompt_forward(x)

    def get_mask_for_x(self, x):
        # calculate size from shape of x
        if len(x.size()) == 4:
            h, w = x.size()[2:4]
            area = h * w
        else:
            area = x.size()[1]

        mask = self.network.mask_dic.get(area, None)
        if mask is None:
            # raise ValueError(f"mask is None for resolution {area}")
            # emb_layers in SDXL doesn't have mask
            # print(f"mask is None for resolution {area}, {x.size()}")
            mask_size = (1, x.size()[1]) if len(x.size()) == 2 else (1, *x.size()[1:-1], 1)
            return torch.ones(mask_size, dtype=x.dtype, device=x.device) / self.network.num_sub_prompts
        if len(x.size()) != 4:
            mask = torch.reshape(mask, (1, -1, 1))
        return mask

    def regional_forward(self, x):
        if "attn2_to_out" in self.lora_name:
            return self.to_out_forward(x)

        if self.network.mask_dic is None:  # sub_prompt_index >= 3
            return self.default_forward(x)

        # apply mask for LoRA result
        lx = self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        mask = self.get_mask_for_x(lx)
        # print("regional", self.lora_name, self.network.sub_prompt_index, lx.size(), mask.size())
        lx = lx * mask

        x = self.org_forward(x)
        x = x + lx

        if "attn2_to_q" in self.lora_name and self.network.is_last_network:
            x = self.postp_to_q(x)

        return x

    def postp_to_q(self, x):
        # repeat x to num_sub_prompts
        has_real_uncond = x.size()[0] // self.network.batch_size == 3
        qc = self.network.batch_size  # uncond
        qc += self.network.batch_size * self.network.num_sub_prompts  # cond
        if has_real_uncond:
            qc += self.network.batch_size  # real_uncond

        query = torch.zeros((qc, x.size()[1], x.size()[2]), device=x.device, dtype=x.dtype)
        query[: self.network.batch_size] = x[: self.network.batch_size]

        for i in range(self.network.batch_size):
            qi = self.network.batch_size + i * self.network.num_sub_prompts
            query[qi : qi + self.network.num_sub_prompts] = x[self.network.batch_size + i]

        if has_real_uncond:
            query[-self.network.batch_size :] = x[-self.network.batch_size :]

        # print("postp_to_q", self.lora_name, x.size(), query.size(), self.network.num_sub_prompts)
        return query

    def sub_prompt_forward(self, x):
        if x.size()[0] == self.network.batch_size:  # if uncond in text_encoder, do not apply LoRA
            return self.org_forward(x)

        emb_idx = self.network.sub_prompt_index
        if not self.text_encoder:
            emb_idx += self.network.batch_size

        # apply sub prompt of X
        lx = x[emb_idx :: self.network.num_sub_prompts]
        lx = self.lora_up(self.lora_down(lx)) * self.multiplier * self.scale

        # print("sub_prompt_forward", self.lora_name, x.size(), lx.size(), emb_idx)

        x = self.org_forward(x)
        x[emb_idx :: self.network.num_sub_prompts] += lx

        return x

    def to_out_forward(self, x):
        # print("to_out_forward", self.lora_name, x.size(), self.network.is_last_network)

        if self.network.is_last_network:
            masks = [None] * self.network.num_sub_prompts
            self.network.shared[self.lora_name] = (None, masks)
        else:
            lx, masks = self.network.shared[self.lora_name]

        # call own LoRA
        x1 = x[self.network.batch_size + self.network.sub_prompt_index :: self.network.num_sub_prompts]
        lx1 = self.lora_up(self.lora_down(x1)) * self.multiplier * self.scale

        if self.network.is_last_network:
            lx = torch.zeros(
                (self.network.num_sub_prompts * self.network.batch_size, *lx1.size()[1:]), device=lx1.device, dtype=lx1.dtype
            )
            self.network.shared[self.lora_name] = (lx, masks)

        # print("to_out_forward", lx.size(), lx1.size(), self.network.sub_prompt_index, self.network.num_sub_prompts)
        lx[self.network.sub_prompt_index :: self.network.num_sub_prompts] += lx1
        masks[self.network.sub_prompt_index] = self.get_mask_for_x(lx1)

        # if not last network, return x and masks
        x = self.org_forward(x)
        if not self.network.is_last_network:
            return x

        lx, masks = self.network.shared.pop(self.lora_name)

        # if last network, combine separated x with mask weighted sum
        has_real_uncond = x.size()[0] // self.network.batch_size == self.network.num_sub_prompts + 2

        out = torch.zeros((self.network.batch_size * (3 if has_real_uncond else 2), *x.size()[1:]), device=x.device, dtype=x.dtype)
        out[: self.network.batch_size] = x[: self.network.batch_size]  # uncond
        if has_real_uncond:
            out[-self.network.batch_size :] = x[-self.network.batch_size :]  # real_uncond

        # print("to_out_forward", self.lora_name, self.network.sub_prompt_index, self.network.num_sub_prompts)
        # if num_sub_prompts > num of LoRAs, fill with zero
        for i in range(len(masks)):
            if masks[i] is None:
                masks[i] = torch.zeros_like(masks[0])

        mask = torch.cat(masks)
        mask_sum = torch.sum(mask, dim=0) + 1e-4
        for i in range(self.network.batch_size):
            # 1枚の画像ごとに処理する
            lx1 = lx[i * self.network.num_sub_prompts : (i + 1) * self.network.num_sub_prompts]
            lx1 = lx1 * mask
            lx1 = torch.sum(lx1, dim=0)

            xi = self.network.batch_size + i * self.network.num_sub_prompts
            x1 = x[xi : xi + self.network.num_sub_prompts]
            x1 = x1 * mask
            x1 = torch.sum(x1, dim=0)
            x1 = x1 / mask_sum

            x1 = x1 + lx1
            out[self.network.batch_size + i] = x1

        # print("to_out_forward", x.size(), out.size(), has_real_uncond)
        return out
"""

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


def create_network(multiplier: float,
                   network_dim: Optional[int],
                   network_alpha: Optional[float],
                   vae: AutoencoderKL,
                   condition_model: Union[CLIPTextModel, List[CLIPTextModel]],
                   unet,
                   neuron_dropout: Optional[float] = None,
                   condition_modality = 'text',
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
    network = LoRANetwork(condition_model=condition_model,
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
                          condition_modality=condition_modality,)

    if up_lr_weight is not None or mid_lr_weight is not None or down_lr_weight is not None:
        network.set_block_lr_weight(up_lr_weight, mid_lr_weight, down_lr_weight)

    return network



# 外部から呼び出す可能性を考慮しておく
from model.lora import LoRANetwork
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
        module_class: Type[object] = LoRAModule,
        varbose: Optional[bool] = False,
        net_key_names: Optional[bool] = False,
        condition_modality='text',
        student_loras : Optional[List] = None, ) -> None:

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
                           student_loras) -> List[LoRAModule]:

            loras = []
            skipped = []
            # prefix ...
            for name, module in root_module.named_modules():

                if module.__class__.__name__ in target_replace_modules:

                    for child_name, child_module in module.named_modules():

                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name # fc1, ...
                            lora_name = lora_name.replace(".", "_")

                            # get student name #
                            student_modules = []
                            for student_lora in student_loras :
                                loras = student_lora.unet_loras + student_lora.image_encoder_loras
                                for lora in loras :
                                    if lora.lora_name == lora_name :
                                        student_modules.append(lora)

                            dim = None
                            alpha = None
                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                # U-Netでblock_dims指定あり
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

                            if block_wise == None :
                                lora = module_class(lora_name,
                                                    child_module,
                                                    self.multiplier,
                                                    dim,
                                                    alpha,
                                                    dropout=dropout,
                                                    rank_dropout=rank_dropout,
                                                    module_dropout=module_dropout,
                                                    student_loras = student_modules)
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
                                                            module_dropout=module_dropout,)
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
                                                     student_loras = student_loras)
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
                                                             student_loras = student_loras)
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
                                                                  prefix=prefix_)
                    self.image_encoder_loras.extend(image_encoder_loras)
                    skipped_ie += skipped
                print(f"create LoRA for Image Encoder : {len(self.image_encoder_loras)} modules.")
                skipped = skipped_ie + skipped_un

                # assertion
                names = set()
                for lora in self.image_encoder_loras + self.unet_loras:
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

    def get_lr_weight(self, lora: LoRAModule) -> float:
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
                params.extend(lora.parameters())
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

def main(args):

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

    network_0_weights_sd = load_file(network_0_state_dict_dir)
    network_1_weights_sd = load_file(network_1_state_dict_dir)
    network_2_weights_sd = load_file(network_2_state_dict_dir) ##########################################################
    network_3_weights_sd = load_file(network_3_state_dict_dir)
    network_4_weights_sd = load_file(network_4_state_dict_dir)
    network_weights = [network_0_weights_sd, network_1_weights_sd, network_2_weights_sd, network_3_weights_sd, network_4_weights_sd]

    print(f' (2) make student networks')

    accelerator = prepare_accelerator(args)
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
    for net_weight in network_weights:
        from model.lora import LoRANetwork
        student_net = LoRANetwork(condition_model=condition_model,
                                  unet=unet,
                                  network_dim=args.network_dim,
                                  network_alpha=args.network_alpha,
                                  neuron_dropout=args.network_dropout,
                                  condition_modality=condition_modality,)
        student_net.load_state_dict(net_weight)
        student_nets.append(student_net)

            
    teacher_network = TeacherLoRANetwork(network_dim=args.network_dim,
                                         network_alpha=args.network_alpha,
                                         condition_model=condition_model,
                                         unet=unet,
                                         neuron_dropout=args.network_dropout,
                                         condition_modality=condition_modality,
                                         student_loras = student_nets,
                                         **net_kwargs, )





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