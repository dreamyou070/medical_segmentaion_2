from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads_3D_4D
from ignite.metrics.confusion_matrix import ConfusionMatrix
import torch
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
import torch.nn.functional as F
def eval_step(engine, batch):
    return batch
@torch.inference_mode()
def evaluation_check(segmentation_head,
                     dataloader,
                     device,
                     condition_model, unet, vae, controller, weight_dtype, epoch,
                     reduction_net, position_embedder, vision_head, positioning_module, args):
    if args.use_segmentation_model :
        segmentation_head.eval()

    with torch.no_grad():
        y_true_list, y_pred_list = [], []

        for global_num, batch in enumerate(dataloader):

            encoder_hidden_states = None  # torch.tensor((1,1,768)).to(device)

            encoder_hidden_states = None  # torch.tensor((1,1,768)).to(device)
            if not args.without_condition:
                if args.use_image_condition:
                    if not args.image_model_training:

                        if args.image_processor == 'pvt':
                            output = condition_model(batch["image_condition"])
                            encoder_hidden_states = vision_head(output)

                        elif args.image_processor == 'vit':
                            with torch.no_grad():
                                output, pix_embedding = condition_model(**batch["image_condition"])
                                encoder_hidden_states = output.last_hidden_state  # [batch, 197, 768]
                                if args.not_use_cls_token:
                                    encoder_hidden_states = encoder_hidden_states[:, 1:, :]
                                if args.only_use_cls_token:
                                    encoder_hidden_states = encoder_hidden_states[:, 0, :]
                    else:
                        with torch.set_grad_enabled(True):
                            if args.image_processor == 'pvt':
                                output = condition_model(batch["image_condition"])
                                # encoder hidden states is dictionary
                                encoder_hidden_states = vision_head(output)
                            elif args.image_processor == 'vit':

                                output, pix_embedding = condition_model(**batch["image_condition"])
                                encoder_hidden_states = output.last_hidden_state  # [batch, 197, 768]
                                if args.not_use_cls_token:
                                    encoder_hidden_states = encoder_hidden_states[:, 1:, :]
                                if args.only_use_cls_token:
                                    encoder_hidden_states = encoder_hidden_states[:, 0, :]

            image = batch['image'].to(dtype=weight_dtype)  # 1,3,512,512
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)  # 1,256*256
            gt = batch['gt'].to(dtype=weight_dtype)  # 1,2,256,256
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor

            # ----------------------------------------------------------------------------------------------------------- #
            # [1] pseudo feature
            # almost impossible
            """
            gt_64_array = batch['gt_64_array'].squeeze() # [1,64,64]
            non_zero_index = torch.nonzero(gt_64_array).flatten()  # class 1 index
            feat = torch.flatten((latents), start_dim=2).squeeze().transpose(1, 0)  # pixel_num, dim [pixel,160]
            anomal_feat = feat[non_zero_index, :]  # [10,4]
            if step == 0:
                generator = DiagonalGaussianDistribution(parameters=anomal_feat, latent_dim=anomal_feat.shape[-1])
            else:
                generator.update(parameters=anomal_feat)
            pseudo_feature = generator.sample(mask_res=args.mask_res, device=device, weight_dtype=weight_dtype) # [1,4,256,256]
            # unet feature generating
            # how to condition ??



            # should unet again ?
            pseudo_masks_pred = segmentation_head.segment_feature(pseudo_feature)  # 1,2,265,265

            pseudo_label = torch.ones_like(batch['gt'])
            pseudo_label[:, 0, :, :] = 0  # all class 1 samples
            pseudo_loss = loss_dicece(input=pseudo_masks_pred,  # [class, 256,256]
                                      target=pseudo_label.to(dtype=weight_dtype, device=accelerator.device))  # [class, 256,256]
            """
            # ----------------------------------------------------------------------------------------------------------- #
            with torch.set_grad_enabled(True):
                if encoder_hidden_states is not None and type(encoder_hidden_states) != dict:
                    if encoder_hidden_states.dim() != 3:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                    if encoder_hidden_states.dim() != 3:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                     noise_type=position_embedder).sample
            query_dict, key_dict = controller.query_dict, controller.key_dict
            controller.reset()
            q_dict = {}
            for layer in args.trg_layer_list:
                query, channel_attn_query = query_dict[layer]  # head, pix_num, dim
                res = int(query.shape[1] ** 0.5)
                if args.test_before_query:
                    query = reshape_batch_dim_to_heads_3D_4D(query)  # 1, res, res, dim
                else:
                    # test after attn (already 1, pix_num, dim)
                    query = query.reshape(1, res, res, -1)
                    query = query.permute(0, 3, 1, 2).contiguous()
                if args.use_positioning_module:
                    query, global_feat = positioning_module(query, layer_name=layer)
                    spatial_attn_query = query
                    if res == 16:
                        global_attn = global_feat
                # channel_attn = [batch, res*res, dim] -> [batch, res, res, dim] ->  [batch, dim, res, res]
                channel_attn_query = channel_attn_query.reshape(1, res, res, -1).permute(0, 3, 1, 2).contiguous()
                # spatial_attn = [batch, dim, res, res]
                # print(f'channel_attn_query = {channel_attn_query.shape} | spatial_attn_query = {spatial_attn_query.shape}')
                q_dict[res] = query
                pred, focus_map = positioning_module.predict_seg(channel_attn_query=channel_attn_query,
                                                                 spatial_attn_query=spatial_attn_query,
                                                                 layer_name=layer,
                                                                 in_map=focus_map)
                # focus_map = [batch, 1, res,res]
                # pred      = [batch, 2, res, res]
                # ------------------------------------------------------------------------------------------------- #
                # mask prediction
            masks_pred = pred
            # ----------------------------------------------------------------------------------------------------------- #
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
    if args.use_segmentation_model :
        segmentation_head.train()
    return IOU_dict, confusion_matrix, dice_coeff