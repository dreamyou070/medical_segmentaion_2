import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads_3D_4D, reshape_batch_dim_to_heads_3D_3D
from utils.loss import dice_loss
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
# pip install pytorch-ignite
from ignite.metrics import Accuracy
from ignite.metrics.confusion_matrix import ConfusionMatrix
import torch
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
import torch.nn.functional as F

def eval_step(engine, batch):
    return batch
@torch.inference_mode()
def evaluation_check(segmentation_head, dataloader, device, text_encoder, unet, vae, controller, weight_dtype,
                     position_embedder, args):
    segmentation_head.eval()

    with torch.no_grad():
        y_true_list, y_pred_list = [], []
        dice_coeff_list = []
        for global_num, batch in enumerate(dataloader):

            # [1] get token
            # torch to list
            key_word_index = batch['key_word_index'][0]

            with torch.set_grad_enabled(True):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
            if args.aggregation_model_d:
                encoder_hidden_states = encoder_hidden_states[:, :args.n_classes, :]
            image = batch['image'].to(dtype=weight_dtype)  # 1,3,512,512
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)  # 1,128*128
            gt = batch['gt'].to(dtype=weight_dtype)  # 1,3,256,256
            gt = gt.permute(0, 2, 3, 1).contiguous()  # .view(-1, gt.shape[-1]).contiguous()   # 1,256,256,3
            gt = gt.view(-1, gt.shape[-1]).contiguous()
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
            with torch.set_grad_enabled(True):
                if args.use_position_embedder:
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                         noise_type=position_embedder)
                else:
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list)
            query_dict, key_dict = controller.query_dict, controller.key_dict
            controller.reset()
            for i, layer in enumerate(args.trg_layer_list):
                query = reshape_batch_dim_to_heads_3D_3D(query_dict[layer][0])  # 1, pix_num, dim
                key = key_dict[layer][0]  # head, pix_num, dim
                attn_map = torch.bmm(query, key.transpose(-1, -2).contiguous())  # 1, pix_num, sen_len
                # key_word_index ---------------------------------------------------------------------------------------
                attn_map = attn_map[:, :, key_word_index]
                # ------------------------------------------------------------------------------------------------------
                original_res = int(attn_map.shape[1] ** 0.5)  # trg_res = 64
                target_res = 64
                upscale_factor = target_res // original_res
                # upscaling
                original_map = attn_map.view(attn_map.shape[0], original_res, original_res, attn_map.shape[2]).permute(
                    0, 3, 1, 2).contiguous()
                attn_map = nn.functional.interpolate(original_map,
                                                     scale_factor=upscale_factor,
                                                     mode='bilinear', align_corners=False)
                if i == 0:
                    attn_maps = attn_map
                else:
                    attn_maps += attn_map
            # attn_maps = [batch, 4, 64, 64] (without segmentation head)
            masks_pred = torch.softmax(attn_maps, dim=1)  # [1,4,256,256]
            if masks_pred.shape[1] < args.n_classes:
                remain_num = args.n_classes - masks_pred.shape[1]
                masks_pred = torch.cat([masks_pred,
                                        torch.zeros(masks_pred.shape[0], remain_num, masks_pred.shape[2],
                                                    masks_pred.shape[3]).to(device=masks_pred.device,
                                                                            dtype=weight_dtype)], dim=1)  # 1,3,64,64
            # upgrading (from 64 to 256)
            masks_pred = segmentation_head(masks_pred)

            #######################################################################################################################
            # [1] pred
            class_num = masks_pred.shape[1]  # 4
            mask_pred_argmax = torch.argmax(masks_pred, dim=1).flatten()  # 256*256
            y_pred_list.append(mask_pred_argmax)
            y_true = gt_flat.squeeze()
            y_true_list.append(y_true)
        #######################################################################################################################
        # [1] pred
        y_pred = torch.cat(y_pred_list).detach().cpu()  # [pixel_num]
        y_pred = F.one_hot(y_pred, num_classes=class_num)  # [pixel_num, C]
        y_true = torch.cat(y_true_list).detach().cpu().long()  # [pixel_num]
        # [2] make confusion engine
        default_evaluator = Engine(eval_step)
        cm = ConfusionMatrix(num_classes=class_num)
        cm.attach(default_evaluator, 'confusion_matrix')
        state = default_evaluator.run([[y_pred, y_true]])
        confusion_matrix = state.metrics['confusion_matrix']
        actual_axis, pred_axis = confusion_matrix.shape
        IOU_dict = {}
        eps = 1e-15
        for actual_idx in range(actual_axis):
            total_actual_num = sum(confusion_matrix[actual_idx])
            total_predict_num = sum(confusion_matrix[:, actual_idx])
            dice_coeff = 2 * confusion_matrix[actual_idx, actual_idx] / (total_actual_num + total_predict_num + eps)
            IOU_dict[actual_idx] = round(dice_coeff.item(), 3)

        # [1] WC Score

    segmentation_head.train()
    return IOU_dict, confusion_matrix, dice_coeff