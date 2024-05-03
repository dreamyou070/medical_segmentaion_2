import torch
import torch.nn.functional as F
import os
from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads_3D_4D
import numpy as np
from PIL import Image
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.engine import *
from data import call_test_dataset
from utils import torch_to_pil
from utils.metrics import generate_Iou, generate_Dice

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
                     boundary_sensitive,
                     accelerator,
                     g_filter_torch,
                     h_filter_torch,
                     args):

        folders = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

        inference_base_dir = os.path.join(args.output_dir, 'inference')
        os.makedirs(inference_base_dir, exist_ok=True)
        epoch_dir = os.path.join(inference_base_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        DSC = 0.0

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

                    pure_path = batch['pure_path'][0]

                    total_loss = 0

                    loss_dict = {}
                    encoder_hidden_states = None  # torch.tensor((1,1,768)).to(device)
                    if not args.without_condition:

                        if args.use_image_condition:

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
                        unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                             noise_type=position_embedder).sample
                    query_dict, key_dict = controller.query_dict, controller.key_dict
                    controller.reset()
                    q_dict = {}

                    for layer in args.trg_layer_list:
                        query = query_dict[layer][0]
                        res = int(query.shape[1] ** 0.5)
                        q_dict[res] = query.reshape(1, res, res, -1).permute(0, 3, 1, 2).contiguous()
                    x16_out, x32_out, x64_out = q_dict[16], q_dict[32], q_dict[64]
                    # ----------------------------------------------------------------------------------------------------------- #
                    # [2] concat feature generating
                    # out_prev, x, feature = segmentation_head.gen_feature(x16_out, x32_out, x64_out) # [batch,2,64,64], [batch,960,64,64], [batch,160,256,256]
                    out_prev, x = segmentation_head.gen_feature(x16_out, x32_out,
                                                                x64_out)  # [batch,2,64,64], [batch,960,64,64], [batch,160,256,256]
                    # ----------------------------------------------------------------------------------------------------------- #
                    # [3] region separation
                    edge_feature = torch.nn.functional.conv2d(out_prev, g_filter_torch, padding=1)  # [batch,1,64,64]
                    region_feature = torch.nn.functional.conv2d(out_prev, h_filter_torch, padding=1)  # [batch,1,64,64]
                    x16_out, x32_out, x64_out = x[:, :320], x[:, 320:640], x[:,640:]  # [batch,320,64,64], [batch,320,64,64], [batch,320,64,64]
                    # ----------------------------------------------------------------------------------------------------------- #
                    # [4] boundary sensitive refinement
                    batch, dim = x16_out.shape[0], x16_out.shape[1]

                    x16_out_edge = (edge_feature * x16_out).view(batch, dim, -1).permute(0, 2,
                                                                                         1).contiguous()  # [batch,len, 320,64,64]
                    x16_out_region = (region_feature * x16_out).view(batch, dim, -1).permute(0, 2, 1).contiguous()

                    x32_out_edge = (edge_feature * x32_out).view(batch, dim, -1).permute(0, 2, 1).contiguous()
                    x32_out_region = (region_feature * x32_out).view(batch, dim, -1).permute(0, 2, 1).contiguous()

                    x64_out_edge = (edge_feature * x64_out).view(batch, dim, -1).permute(0, 2, 1).contiguous()
                    x64_out_region = (region_feature * x64_out).view(batch, dim, -1).permute(0, 2, 1).contiguous()

                    masks_pred = boundary_sensitive([x16_out_edge, x32_out_edge, x64_out_edge],
                                                    [x16_out_region, x32_out_region, x64_out_region],
                                                    [x16_out, x32_out, x64_out])  # 1,2,64,64

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
                    DSC = DSC + dice
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
                         #########################

                        total_img.save(os.path.join(save_base_dir, f'{pure_path}'))

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

            # [1] making confusion matrix
            confusion_matrix = state.metrics['confusion_matrix']
            Iou = generate_Iou(confusion_matrix)
            Dice = generate_Dice(confusion_matrix)
            mean_Iou = torch.mean(Iou)
            mean_Dice = torch.mean(Dice)

            # [2] saving
            if accelerator.is_main_process:
                print(f' {_data_name} finished !')
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
                    # [1] Iou score
                    for i, score in enumerate(Iou):
                        f.write(f'class {i} IoU Score = {score} ')
                        print(f'class {i} IoU Score = {score} ')
                    f.write(f'| mean IoU Score = {mean_Iou} \n')
                    print(f'| mean IoU Score = {mean_Iou} \n')
                    # [2] Dice score
                    for i, score in enumerate(Dice):
                        f.write(f'class {i} Dice Score = {score} ')
                        print(f'class {i} Dice Score = {score} ')
                    f.write(f'| mean Dice Score = {mean_Dice} \n')
                    print(f'| mean Dice Score = {mean_Dice} \n')
            segmentation_head.train()