import torch
import torch.nn.functional as F
import os
import argparse
from model import call_model_package
from model.pe import AllPositionalEmbedding
from utils.accelerator_utils import prepare_accelerator
from model.segmentation_unet import SemanticModel
from model.reduction_model import ReductionNet
from data.dataloader import TestDataset
from torch import nn
from attention_store import AttentionStore
from utils.attention_control import passing_argument, register_attention_control
from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads_3D_4D, reshape_batch_dim_to_heads_3D_3D
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.engine import *
from torchvision import transforms
def eval_step(engine, batch):
    return batch
def torch_to_pil(torch_img):
    # torch_img = [3, H, W], from -1 to 1
    if torch_img.dim() == 3 :
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
    else :
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8)
    pil = Image.fromarray(np_img).convert("RGB")
    return pil

def main(args):

    print(f' step 0. accelerator')
    args.logging_dir = os.path.join(args.output_dir, 'log_infer')
    os.makedirs(args.logging_dir, exist_ok=True)
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f' step 1. make model')
    print(f' (1) stable diffusion model')
    weight_dtype, save_dtype = prepare_dtype(args)
    condition_model, vae, unet, network, _ = call_model_package(args, weight_dtype, accelerator)
    print(f' (2) lora network and loading model')
    network.to(dtype=weight_dtype, device=accelerator.device)
    print(f' (3) segmentation head and loading pretrained')
    segmentation_head = SemanticModel(n_classes=args.n_classes,
                                      mask_res=args.mask_res,
                                      use_layer_norm=args.use_layer_norm,)
    # pt file
    segmentation_state_dict = torch.load(args.segmentation_head_weights)
    segmentation_head.load_state_dict(segmentation_state_dict)
    segmentation_head.to(dtype=weight_dtype, device=accelerator.device)
    pure_path = os.path.split(args.segmentation_head_weights)[-1]
    pure_path = os.path.splitext(pure_path)[0]
    num = pure_path.split('-')[-1]
    print(f' (4) reduction_net')
    reduction_net = None
    if args.reducing_redundancy:
        reduction_net = ReductionNet(cross_dim=768,
                                     class_num=args.n_classes)
        # loading
        reduction_folder = os.path.join(args.output_dir, 'reduction_net')
        reduction_file = os.path.join(reduction_folder, f'reduction-{num}.pt')
        reduction_net.load_state_dict(torch.load(reduction_file))
        reduction_net.to(dtype=weight_dtype, device=accelerator.device)

    print(f' (5) position_embedder')
    position_embedder = None
    if args.use_position_embedder:
        position_embedder = AllPositionalEmbedding()
        # loading
        position_embedder_folder = os.path.join(args.output_dir, 'position_embedder')
        position_embedder_file = os.path.join(position_embedder_folder, f'position_embedder-{num}.pt')
        position_embedder.load_state_dict(torch.load(position_embedder_file))
        position_embedder.to(dtype=weight_dtype, device=accelerator.device)

    print(f' (5) make controller')
    controller = AttentionStore()
    register_attention_control(unet, controller)

    # [2] image model
    if args.image_processor == 'clip':
        condition_transform = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif args.image_processor == 'vit':
        # ViTModel
        condition_transform = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    elif args.image_processor == 'pvt':
        condition_transform = transforms.Compose([transforms.Resize((384,384)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])



    print(f' step 2. check data path')
    if not args.inference_with_training_data :
        save_base = os.path.join(args.output_dir, 'thesis_output')
        os.makedirs(save_base, exist_ok=True)
        for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

            # [1] data_path here
            data_path = os.path.join(args.base_path, _data_name)

            # [2] save_path
            save_base_dir = os.path.join(save_base, _data_name)
            os.makedirs(save_base_dir, exist_ok=True)

            image_root = os.path.join(data_path, 'images')
            gt_root = os.path.join(data_path, 'masks')

            test_dataset = TestDataset(image_root = image_root,
                                       gt_root = gt_root,
                                       resize_shape = (512,512),
                                       image_processor = condition_transform,
                                       latent_res = 64,
                                       n_classes = 4,
                                       mask_res = 256,
                                       use_data_aug=False,)
            test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=1,
                                                          shuffle=False)
            test_dataloader = accelerator.prepare(test_dataloader)
            # [4] output dir
            device = accelerator.device
            with torch.no_grad():
                y_true_list, y_pred_list = [], []
                for global_num, batch in enumerate(test_dataloader):
                    if not args.without_condition:
                        if args.use_image_condition:
                            """ condition model is already on device and dtype """
                            with torch.no_grad():
                                output, pix_embedding = condition_model(**batch["image_condition"])
                                encoder_hidden_states = output.last_hidden_state  # [batch, 197, 768]
                                if args.not_use_cls_token:
                                    encoder_hidden_states = encoder_hidden_states[:, 1:, :]
                                if args.only_use_cls_token:
                                    encoder_hidden_states = encoder_hidden_states[:, 0, :]
                                if args.reducing_redundancy:
                                    encoder_hidden_states = reduction_net(encoder_hidden_states)

                    if args.use_text_condition:
                        with torch.set_grad_enabled(True):
                            encoder_hidden_states = condition_model(batch["input_ids"].to(device))[
                                "last_hidden_state"]  # [batch, 77, 768]
                    # [1] original image (1,3,512,512)
                    image = batch['image'].to(dtype=weight_dtype)

                    # [2] gt image
                    gt_flat = batch['gt_flat'].to(dtype=weight_dtype)  # 1,128*128
                    gt = batch['gt'].to(dtype=weight_dtype)  # 1,3,256,256
                    #gt = gt.permute(0, 2, 3, 1).contiguous()  # .view(-1, gt.shape[-1]).contiguous()   # 1,256,256,3
                    #gt = gt.view(-1, gt.shape[-1]).contiguous()
                    with torch.no_grad():
                        latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
                    with torch.set_grad_enabled(True):
                        if encoder_hidden_states.dim() != 3:
                            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                        if encoder_hidden_states.dim() != 3:
                            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                        noise_pred = unet(latents, 0, encoder_hidden_states,
                                              trg_layer_list=args.trg_layer_list).sample
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
                    # masks_pred = [batch, 2, 256,256]

                    r = int(mask_pred_argmax.shape[0] ** .5)
                    y_pred_list.append(mask_pred_argmax)
                    y_true = gt_flat.squeeze()
                    y_true_list.append(y_true)

                    # [2] saving image (all in 256 X 256)
                    original_pil = torch_to_pil(image.squeeze().detach().cpu()).resize((r,r))
                    gt_pil = torch_to_pil(gt.squeeze().detach().cpu()).resize((r,r))

                    predict_pil = torch_to_pil(mask_pred_argmax.reshape((r,r)).contiguous().detach().cpu())
                    merged_pil = Image.blend(original_pil, predict_pil, 0.4)
                    total_img = Image.new('RGB', (r * 4, r))
                    total_img.paste(original_pil, (0, 0))
                    total_img.paste(gt_pil, (r, 0))
                    total_img.paste(predict_pil, (r * 2, 0))
                    total_img.paste(merged_pil, (r * 3, 0))
                    pure_path = batch['pure_path'][0]
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
            print(f' {_data_name} finished !')
            # saving score
            # saving
            if is_main_process:
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
    else :

        save_base = os.path.join(args.output_dir, 'inference_with_training_data')
        os.makedirs(save_base, exist_ok=True)
        # [1] data_path here
        data_path = os.path.join(args.base_path, 'res_256')
        # [2] save_path

        image_root = os.path.join(data_path, 'image_256')
        gt_root = os.path.join(data_path, 'mask_256')

        test_dataset = TestDataset(image_root=image_root,
                                   gt_root=gt_root,
                                   resize_shape=(512, 512),
                                   image_processor=condition_transform,
                                   latent_res=64,
                                   n_classes=4,
                                   mask_res=256,
                                   use_data_aug=False, )
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1,
                                                      shuffle=False)
        test_dataloader = accelerator.prepare(test_dataloader)
        # [4] output dir
        device = accelerator.device
        with torch.no_grad():
            y_true_list, y_pred_list = [], []
            for global_num, batch in enumerate(test_dataloader):
                if not args.without_condition:
                    if args.use_image_condition:
                        """ condition model is already on device and dtype """
                        with torch.no_grad():
                            output, pix_embedding = condition_model(**batch["image_condition"])
                            encoder_hidden_states = output.last_hidden_state  # [batch, 197, 768]
                            if args.not_use_cls_token:
                                encoder_hidden_states = encoder_hidden_states[:, 1:, :]
                            if args.only_use_cls_token:
                                encoder_hidden_states = encoder_hidden_states[:, 0, :]
                            if args.reducing_redundancy:
                                encoder_hidden_states = reduction_net(encoder_hidden_states)

                if args.use_text_condition:
                    with torch.set_grad_enabled(True):
                        encoder_hidden_states = condition_model(batch["input_ids"].to(device))[
                            "last_hidden_state"]  # [batch, 77, 768]
                # [1] original image (1,3,512,512)
                image = batch['image'].to(dtype=weight_dtype)

                # [2] gt image
                gt_flat = batch['gt_flat'].to(dtype=weight_dtype)  # 1,128*128
                gt = batch['gt'].to(dtype=weight_dtype)  # 1,3,256,256
                # gt = gt.permute(0, 2, 3, 1).contiguous()  # .view(-1, gt.shape[-1]).contiguous()   # 1,256,256,3
                # gt = gt.view(-1, gt.shape[-1]).contiguous()
                with torch.no_grad():
                    latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
                with torch.set_grad_enabled(True):
                    if encoder_hidden_states.dim() != 3:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                    if encoder_hidden_states.dim() != 3:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                    noise_pred = unet(latents, 0, encoder_hidden_states,
                                      trg_layer_list=args.trg_layer_list).sample
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
                # masks_pred = [batch, 2, 256,256]

                r = int(mask_pred_argmax.shape[0] ** .5)
                y_pred_list.append(mask_pred_argmax)
                y_true = gt_flat.squeeze()
                y_true_list.append(y_true)

                # [2] saving image (all in 256 X 256)
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
                total_img.save(os.path.join(save_base, f'{pure_path}'))

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
        # saving score
        # saving
        if is_main_process:
            print(f'  - precision dictionary = {IOU_dict}')
            print(f'  - confusion_matrix = {confusion_matrix}')
            confusion_matrix = confusion_matrix.tolist()
            confusion_save_dir = os.path.join(save_base, 'confusion.txt')
            with open(confusion_save_dir, 'a') as f:
                for i in range(len(confusion_matrix)):
                    for j in range(len(confusion_matrix[i])):
                        f.write(' ' + str(confusion_matrix[i][j]) + ' ')
                    f.write('\n')
                f.write('\n')

            score_save_dir = os.path.join(save_base, 'score.txt')
            with open(score_save_dir, 'a') as f:
                dices = []
                for k in IOU_dict:
                    dice = float(IOU_dict[k])
                    f.write(f'class {k} = {dice} ')
                    dices.append(dice)
                dice_coeff = sum(dices) / len(dices)
                f.write(f'| dice_coeff = {dice_coeff}')
                f.write(f'\n')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/Pranet/test')
    parser.add_argument('--save_base', type=str, default='./result_sy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    # step 2. dataset
    parser.add_argument("--resize_shape", type=int, default=512)
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
    parser.add_argument("--aggregation_model_a", action='store_true')
    parser.add_argument("--aggregation_model_b", action='store_true')
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
    parser.add_argument("--text_before_query", action='store_true')
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
    parser.add_argument("--use_layer_norm", action='store_true')
    parser.add_argument("--original_learning", action='store_true')
    parser.add_argument("--segmentation_head_weights", type= str)
    parser.add_argument("--inference_with_training_data", action='store_true')
    parser.add_argument("--use_position_embedding", action='store_true')
    args = parser.parse_args()
    passing_argument(args)
    from data.dataloader import passing_mvtec_argument
    passing_mvtec_argument(args)
    main(args)
    args = parser.parse_args()
    main(args)