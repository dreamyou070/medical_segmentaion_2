import torch
import torch.nn.functional as F
import os
from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads_3D_4D
import numpy as np
from PIL import Image
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.engine import *
from data import call_test_dataset

def eval_step(engine, batch):
    return batch

def torch_to_pil(torch_img):
    # torch_img = [3, H, W], from -1 to 1
    if torch_img.dim() == 3:
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
    else:
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8)
    pil = Image.fromarray(np_img).convert("RGB")
    return pil

def eval_step(engine, batch):
    return batch

@torch.inference_mode()
def evaluation_check(segmentation_head,
                     condition_model,
                     unet,
                     vae,
                     controller,
                     weight_dtype,
                     epoch,
                     position_embedder,
                     vision_head,
                     self_feature_merger,
                     accelerator,
                     args):

        folders = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

        inference_base_dir = os.path.join(args.output_dir, 'inference')
        os.makedirs(inference_base_dir, exist_ok=True)
        epoch_dir = os.path.join(inference_base_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        DSC = 0.0
        IOU = 0.0
        for _data_name in folders:

            # [1] data_path here
            save_base_dir = os.path.join(epoch_dir, _data_name)
            os.makedirs(save_base_dir, exist_ok=True)

            # [2] dataloader
            test_dataloader = call_test_dataset(args, _data_name)
            test_dataloader = accelerator.prepare(test_dataloader)

            with torch.no_grad():

                y_true_list, y_pred_list = [], []

                for step, batch in enumerate(test_dataloader):
                    total_loss = 0
                    focus_map = None
                    loss_dict = {}
                    encoder_hidden_states = None  # torch.tensor((1,1,768)).to(device)
                    if not args.without_condition:

                        if args.use_image_condition:

                            if not args.image_model_training:
                                if args.image_processor == 'pvt':
                                    output = condition_model(batch["image_condition"])
                                    encoder_hidden_states = vision_head(output)
                                elif args.image_processor == 'vit':
                                    output, pix_embedding = condition_model(**batch["image_condition"])
                                    encoder_hidden_states = output.last_hidden_state  # [batch, 197, 768]
                            else:
                                with torch.set_grad_enabled(True):
                                    if args.image_processor == 'pvt':
                                        output = condition_model(batch["image_condition"])
                                        encoder_hidden_states = vision_head(output)
                                    elif args.image_processor == 'vit':
                                        output, pix_embedding = condition_model(**batch["image_condition"])
                                        encoder_hidden_states = output.last_hidden_state  # [batch, 197, 768]
                    image = batch['image'].to(dtype=weight_dtype)  # 1,3,512,512
                    gt_flat = batch['gt_flat'].to(dtype=weight_dtype)  # 1,256*256
                    gt = batch['gt'].to(dtype=weight_dtype)  # 1,2,256,256
                    with torch.no_grad():
                        latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
                    # ----------------------------------------------------------------------------------------------------------- #
                    with torch.set_grad_enabled(True):
                        if encoder_hidden_states is not None and type(encoder_hidden_states) != dict:
                            if encoder_hidden_states.dim() != 3:
                                encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                            if encoder_hidden_states.dim() != 3:
                                encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                        unet(latents,
                             0,
                             encoder_hidden_states,
                             trg_layer_list=[args.trg_layer_list, args.trg_layers],
                             noise_type=[position_embedder, self_feature_merger]).sample
                    feature = controller.query_list[0]  # 1, 64*64, 320
                    query_dict, key_dict = controller.query_dict, controller.key_dict
                    controller.reset()
                    q_dict = {}
                    global_feature = None
                    for layer in args.trg_layers:
                        query = query_dict[layer][0]  # head, pix_num, dim
                        res = int(query.shape[1] ** 0.5)
                        query = query.reshape(1, res, res, -1)
                        query = query.permute(0, 3, 1, 2).contiguous()
                        q_dict[res] = query

                    x16_out, x32_out, x64_out = q_dict[16], q_dict[32], q_dict[64]
                    _, features = segmentation_head.gen_feature(x16_out, x32_out, x64_out)  # [1,4,256,256]
                    masks_pred = segmentation_head.segment_feature(features)  # here problem
                    masks_pred_ = masks_pred.permute(0, 2, 3, 1).contiguous().view(-1, masks_pred.shape[-1]).contiguous()

                    # [1] pred
                    class_num = masks_pred.shape[1]  # 4
                    mask_pred_argmax = torch.argmax(masks_pred, dim=1).flatten()  # 256*256
                    # masks_pred = [batch, 2, 256,256]
                    r = int(mask_pred_argmax.shape[0] ** .5)
                    y_pred_list.append(mask_pred_argmax)
                    y_true = gt_flat.squeeze()
                    y_true_list.append(y_true)

                    smooth = 1
                    target = y_true
                    input = mask_pred_argmax
                    input_flat = input
                    target_flat = target
                    intersection = (input_flat * target_flat)  # only both are class 1
                    dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)  # dice = every pixel by pixel
                    dice = float(dice)
                    iou =  (intersection.sum() + smooth) / (input.sum() + target.sum() - intersection.sum() + smooth)
                    DSC = DSC + dice
                    IOU = IOU + iou
                    # [2] saving image (all in 256 X 256)
                    if args.save_image :
                        original_pil = torch_to_pil(image.squeeze().detach().cpu()).resize((r, r))
                        gt_pil = torch_to_pil(gt.squeeze().detach().cpu()).resize((r, r))

                        predict_pil = torch_to_pil(mask_pred_argmax.reshape((r, r)).contiguous().detach().cpu())
                        merged_pil = Image.blend(original_pil, predict_pil, 0.4)
                        total_img = Image.new('RGB', (r * 4, r))
                        total_img.paste(original_pil, (0, 0))
                        total_img.paste(gt_pil, (r, 0))
                        total_img.paste(predict_pil, (r * 2, 0))
                        total_img.paste(merged_pil, (r * 3, 0))
                        pure_path = batch['pure_path'][0]

                        total_img.save(os.path.join(save_base_dir, f'{pure_path}'))

                #######################################################################################################################
                #######################################################################################################################
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
                    dice_coeff = 2 * confusion_matrix[actual_idx, actual_idx] / (
                                total_actual_num + total_predict_num + eps)
                    IOU_dict[actual_idx] = round(dice_coeff.item(), 3)
                # [1] WC Score
                if accelerator.is_main_process:
                    dataset_dice = DSC / len(folders)
                    print(f' {_data_name} finished !')
                    print(f'  - precision dictionary = {IOU_dict}')
                    print(f'  - confusion_matrix = {confusion_matrix}')
                    confusion_matrix = confusion_matrix.tolist()
                    confusion_save_dir = os.path.join(save_base_dir, 'confusion.txt')
                    with open(confusion_save_dir, 'a') as f:
                        f.write(f' data_name = {_data_name} \n')
                        for i in range(len(confusion_matrix)):
                            for j in range(len(confusion_matrix[i])):
                                f.write(' ' + str(confusion_matrix[i][j]) + ' ')
                            f.write('\n')
                        f.write('\n')

                    score_save_dir = os.path.join(save_base_dir, 'score.txt')
                    with open(score_save_dir, 'a') as f:
                        dices = []
                        f.write(f' data_name = {_data_name} | ')
                        for k in IOU_dict:
                            dice = float(IOU_dict[k])
                            f.write(f'class {k} = {dice} ')
                            dices.append(dice)
                        dice_coeff = sum(dices) / len(dices)
                        f.write(f'| dice_coeff = {dice_coeff}')
                        f.write(f'\n')
                        f.write(f' dice score = {dataset_dice} \n')
                        f.write(f' IOU score = {IOU} \n')
                segmentation_head.train()