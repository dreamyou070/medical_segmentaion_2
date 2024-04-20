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
import matplotlib.pyplot as plt
from PIL import Image
import os
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution # from diffusers
def eval_step(engine, batch):
    return batch
@torch.inference_mode()
def evaluation_check(segmentation_head, dataloader, device,
                     condition_model, unet, vae, controller, weight_dtype, epoch,
                     simple_linear, position_embedder, args):

    segmentation_head.eval()

    with torch.no_grad():
        y_true_list, y_pred_list = [], []
        dice_coeff_list = []

        for global_num, batch in enumerate(dataloader):

            loss_dict = {}
            if args.use_image_condition:
                if not args.image_model_training:
                    with torch.no_grad():
                        cond_input = batch["image_condition"].data["pixel_values"]  # pixel_value = [3, 224,224]
                        if args.image_processor == 'clip':
                            encoder_hidden_states = condition_model.get_image_features(
                                **batch["image_condition"])  # [Batch, 1, 768]
                            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                        elif args.image_processor == 'vit':
                            encoder_hidden_states = condition_model(
                                **batch["image_condition"]).last_hidden_state  # [batch, 197, 768]
                else:
                    with torch.set_grad_enabled(True):
                        cond_input = batch["image_condition"].data["pixel_values"]  # pixel_value = [batch,3,224,224]
                        if args.image_processor == 'clip':
                            encoder_hidden_states = condition_model.get_image_features(
                                **batch["image_condition"])  # [Batch, 1, 768]
                            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                        elif args.image_processor == 'vit':
                            img_con = batch["image_condition"]
                            encoder_hidden_states = condition_model(
                                **batch["image_condition"].to(device)).last_hidden_state  # [batch, 197, 768]
            if args.use_text_condition:
                with torch.set_grad_enabled(True):
                    encoder_hidden_states = condition_model(batch["input_ids"].to(device))[
                        "last_hidden_state"]  # [batch, 77, 768]
            image = batch['image'].to(dtype=weight_dtype)  # 1,3,512,512
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)  # 1,128*128
            gt = batch['gt'].to(dtype=weight_dtype)  # 1,3,256,256
            gt = gt.permute(0, 2, 3, 1).contiguous()  # .view(-1, gt.shape[-1]).contiguous()   # 1,256,256,3
            gt = gt.view(-1, gt.shape[-1]).contiguous()
            # key_word_index = batch['key_word_index'][0] # torch([10,14])
            # target key word should intense
            # how can i increase the alignment between image and text ?

            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor

            with torch.set_grad_enabled(True):
                unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list)

            query_dict, key_dict = controller.query_dict, controller.key_dict
            controller.reset()
            q_dict = {}
            for layer in args.trg_layer_list:
                query = query_dict[layer][0]  # head, pix_num, dim
                res = int(query.shape[1] ** 0.5)
                if args.text_before_query:
                    query = reshape_batch_dim_to_heads_3D_4D(query)  # 1, res, res, dim
                else:
                    query = query.reshape(1, res, res, -1)
                    query = query.permute(0, 3, 1, 2).contiguous()
                q_dict[res] = query

            x16_out, x32_out, x64_out = q_dict[16], q_dict[32], q_dict[64]

            masks_pred = segmentation_head(x16_out, x32_out, x64_out)  # [1,4,256,256]




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