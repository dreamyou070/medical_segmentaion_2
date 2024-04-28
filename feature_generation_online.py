""" Context-aware Network """
# 일부러 전략을 세워가면서 하자.
# 1.1. extracting edge feature
# 1.2. deep features are extracted according to edge feature
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


def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

# image conditioned segmentation mask generating

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

    print(f'\n step 3. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    condition_model, vae, unet, network, condition_modality = call_model_package(args, weight_dtype, accelerator)
    segmentation_head = None
    if args.use_segmentation_model :
        if args.use_simple_segmodel :
            segmentation_head = SemanticModel(n_classes=args.n_classes,
                                              mask_res=args.mask_res,
                                              use_layer_norm = args.use_layer_norm)
        else :
            segmentation_head = PFNet(n_classes=args.n_classes, mask_res=args.mask_res,
                                      use_layer_norm = args.use_layer_norm)

    vision_head = None
    if args.image_processor == 'pvt' :
        vision_head = vision_condition_head(reverse = args.reverse)
    position_embedder = None
    if args.use_position_embedder :
        position_embedder = AllPositionalEmbedding()
    positioning_module = None
    if args.use_positioning_module :
        positioning_module = AllPositioning(use_channel_attn=args.use_channel_attn,
                                            use_self_attn=args.use_self_attn,
                                            n_classes = args.n_classes,)

    print(f'\n step 4. dataset and dataloader')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader, test_dataloader, tokenizer = call_dataset(args)

    print(f'\n step 5. optimizer')
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr,
                                                        args.learning_rate, condition_modality=condition_modality,)
    if args.use_position_embedder :
        trainable_params.append({"params": position_embedder.parameters(), "lr": args.learning_rate})
    if args.image_processor == 'pvt' :
        trainable_params.append({"params": vision_head.parameters(), "lr": args.learning_rate})
    if args.use_positioning_module:
        trainable_params.append({"params": positioning_module.parameters(), "lr": args.learning_rate})
    if args.use_segmentation_model :
        trainable_params.append({"params": segmentation_head.parameters(), "lr": args.learning_rate})
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)

    print(f'\n step 6. lr')
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 7. loss function')
    loss_CE = nn.CrossEntropyLoss()
    loss_FC = Multiclass_FocalLoss()
    loss_dicece = DiceCELoss(include_background=False,
                             to_onehot_y=False,
                             sigmoid=False,
                             softmax=True,
                             squared_pred=True,
                             lambda_dice=args.dice_weight,
                             smooth_nr=1e-5,
                             smooth_dr=1e-5,
                             weight=None, )

    print(f'\n step 8. model to device')
    condition_model = accelerator.prepare(condition_model)
    condition_models = transform_models_if_DDP([condition_model])
    segmentation_head, unet, network, optimizer, train_dataloader, test_dataloader, lr_scheduler = \
      accelerator.prepare(segmentation_head, unet, network, optimizer, train_dataloader, test_dataloader, lr_scheduler)
    if args.use_positioning_module:
        positioning_module = accelerator.prepare(positioning_module)
    if args.use_position_embedder :
        position_embedder = accelerator.prepare(position_embedder)
        position_embedder = transform_models_if_DDP([position_embedder])[0]
    if args.image_processor == 'pvt' :
        vision_head = accelerator.prepare(vision_head)
        vision_head = transform_models_if_DDP([vision_head])[0]
    unet, network = transform_models_if_DDP([unet, network])
    if args.use_segmentation_model:
        segmentation_head = transform_models_if_DDP([segmentation_head])[0]
    if args.gradient_checkpointing:
        unet.train()
        if args.use_segmentation_model:
            segmentation_head.train()
        for t_enc in condition_models:
            t_enc.train()
            if args.train_text_encoder:
                t_enc.text_model.embeddings.requires_grad_(True)
    else:
        unet.eval()
        for t_enc in condition_models:
            t_enc.eval()
        del t_enc
        network.prepare_grad_etc()

    print(f'\n step 9. registering saving tensor')
    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f'\n step 10. Training !')
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []
    for epoch in range(args.start_epoch, args.max_train_epochs):

        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")
        epoch_loss_total =0

        for step, batch in enumerate(train_dataloader):
            total_loss = 0
            focus_map = None
            device = accelerator.device
            loss_dict = {}
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
                                encoder_hidden_states = vision_head(output)
                            elif args.image_processor == 'vit':
                                output, pix_embedding = condition_model(**batch["image_condition"])
                                encoder_hidden_states = output.last_hidden_state  # [batch, 197, 768]
                                if args.not_use_cls_token:
                                    encoder_hidden_states = encoder_hidden_states[:, 1:, :]
                                if args.only_use_cls_token:
                                    encoder_hidden_states = encoder_hidden_states[:, 0, :]
            image = batch['image'].to(dtype=weight_dtype)      # 1,3,512,512
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)  # 1,256*256
            gt = batch['gt'].to(dtype=weight_dtype)            # 1,2,256,256
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
                if encoder_hidden_states is not None and type(encoder_hidden_states) != dict :
                    if encoder_hidden_states.dim() != 3:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                    if encoder_hidden_states.dim() != 3:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type = position_embedder).sample
            query_dict, key_dict = controller.query_dict, controller.key_dict
            controller.reset()
            q_dict = {}
            for layer in args.trg_layer_list:
                # self attention is too week ?
                # didnot use spatial attention ...
                # spatial query
                query, channel_attn_query = query_dict[layer]  # head, pix_num, dim
                res = int(query.shape[1] ** 0.5)
                #if args.test_before_query:
                #    query = reshape_batch_dim_to_heads_3D_4D(query)  # 1, res, res, dim
                #else:
                query = query.reshape(1, res, res, -1)
                query = query.permute(0, 3, 1, 2).contiguous()   # 1, dim, res, res
                spatial_attn_query = query
                if args.use_positioning_module :
                    # spatial
                    spatial_attn_query, global_feat = positioning_module(query, layer_name=layer)
                channel_attn_query = channel_attn_query.reshape(1, res, res, -1).permute(0, 3, 1, 2).contiguous()
                # channel_attn_query = [1, 320, 64, 64]
                # spatial_attn_query = [1, 320, 64, 64]
                if focus_map is None :
                    if args.use_max_for_focus_map :
                        focus_map = torch.max(channel_attn_query, dim=1, keepdim=True).values
                    else :
                        focus_map = torch.mean(channel_attn_query, dim=1, keepdim=True)
                pred, feature, focus_map = positioning_module.predict_seg(channel_attn_query=channel_attn_query,
                                                                          spatial_attn_query=spatial_attn_query,
                                                                          layer_name=layer,
                                                                          in_map=focus_map)
                #pred = channel_attn_query
                q_dict[res] = feature
                # focus_map = [batch, 1, res,res]
                # pred      = [batch, 2, res, res]
                # ------------------------------------------------------------------------------------------------- #
                # mask prediction
                total_loss += loss_dicece(input = pred,  # [class, 256,256]
                                   target= batch['res_array_gt'][str(res)].to(dtype=weight_dtype)).mean()
            x16_out, x32_out, x64_out = q_dict[16], q_dict[32], q_dict[64]
            if args.use_simple_segmodel :
                _, features = segmentation_head.gen_feature(x16_out, x32_out, x64_out)  # [1,160,256,256]
                # [2] segmentation head
                masks_pred = segmentation_head.segment_feature(features)  # [1,2,  256,256]  # [1,160,256,256]
            else :
                features = segmentation_head.gen_feature(x16_out, x32_out, x64_out, global_feat)  # [1,160,256,256]
                # [2] segmentation head
                masks_pred = segmentation_head.segment_feature(features)  # [1,2,  256,256]
            masks_pred_ = masks_pred.permute(0, 2, 3, 1).contiguous().view(-1, masks_pred.shape[-1]).contiguous()
            if args.use_dice_ce_loss:
                loss = loss_dicece(input=masks_pred,                           # [class, 256,256]
                                   target=batch['gt'].to(dtype=weight_dtype)) #  [class, 256,256]
            else:
                loss = loss_CE(masks_pred_, gt_flat.squeeze().to(torch.long))  # 128*128
                loss_dict['cross_entropy_loss'] = loss.item()
                # [5.2] Focal Loss
                focal_loss = loss_FC(masks_pred_, gt_flat.squeeze().to(masks_pred.device))  # N
                if args.use_monai_focal_loss: focal_loss = focal_loss.mean()
                loss += focal_loss
                loss_dict['focal_loss'] = focal_loss.item()
                loss = loss.mean()
            # ----------------------------------------------------------------------------------------------------------- #
            # why does it takes so much time ... ?
            # is there any other way to accelerate this process ?
            # [3]
            #if args.online_pseudo_loss :
                #if args.anomal_small_loss:
                #    loss = loss + pseudo_loss * args.pseudo_loss_weight + anomal_loss * args.anomal_loss_weight
                #else :
                #loss = loss + pseudo_loss * args.pseudo_loss_weight

            total_loss += loss.mean()
            current_loss = total_loss.detach().item()

            if epoch == args.start_epoch:
                loss_list.append(current_loss)
            else:
                epoch_loss_total -= loss_list[step]
                loss_list[step] = current_loss
            epoch_loss_total += current_loss
            avr_loss = epoch_loss_total / len(loss_list)
            loss_dict['avr_loss'] = avr_loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if is_main_process:
                progress_bar.set_postfix(**loss_dict)
            if global_step >= args.max_train_steps:
                break
        # ----------------------------------------------------------------------------------------------------------- #
        accelerator.wait_for_everyone()
        if is_main_process:
            saving_epoch = str(epoch + 1).zfill(6)
            save_model(args,
                       saving_folder='model',
                       saving_name=f'lora-{saving_epoch}.safetensors',
                       unwrapped_nw=accelerator.unwrap_model(network),
                       save_dtype=save_dtype)
            if args.use_segmentation_model :
                save_model(args,
                           saving_folder='segmentation',
                           saving_name=f'segmentation-{saving_epoch}.pt',
                           unwrapped_nw=accelerator.unwrap_model(segmentation_head),
                           save_dtype=save_dtype)

            if args.use_position_embedder :
                save_model(args,
                           saving_folder='position_embedder',
                           saving_name=f'position-{saving_epoch}.pt',
                           unwrapped_nw=accelerator.unwrap_model(position_embedder),
                           save_dtype=save_dtype)
            if args.image_processor == 'pvt' :
                save_model(args,
                           saving_folder='vision_head',
                           saving_name=f'vision-{saving_epoch}.pt',
                           unwrapped_nw=accelerator.unwrap_model(vision_head),
                           save_dtype=save_dtype)
            if args.use_position_embedder :
                save_model(args,
                            saving_folder='positioning_module',
                            saving_name=f'positioning-{saving_epoch}.pt',
                            unwrapped_nw=accelerator.unwrap_model(positioning_module),
                            save_dtype=save_dtype)

            """
            save_model(args,
                       saving_folder='anomal_generator',
                       saving_name=f'anomal-{saving_epoch}.pt',
                       unwrapped_nw=accelerator.unwrap_model(anomal_generator),
                       save_dtype=save_dtype)
            """

        # ----------------------------------------------------------------------------------------------------------- #
        # [7] evaluate
        loader = test_dataloader
        if args.check_training:
            print(f'test with training data')
            loader = train_dataloader

        score_dict, confusion_matrix, _ = evaluation_check(segmentation_head,
                                                           loader,
                                                           accelerator.device,
                                                           condition_model,
                                                           unet, vae, controller, weight_dtype, epoch,
                                                           None,
                                                           position_embedder, vision_head,
                                                           positioning_module,
                                                           args)

        # saving
        if is_main_process:
            print(f'  - precision dictionary = {score_dict}')
            print(f'  - confusion_matrix = {confusion_matrix}')
            confusion_matrix = confusion_matrix.tolist()
            confusion_save_dir = os.path.join(args.output_dir, 'confusion.txt')
            with open(confusion_save_dir, 'a') as f:
                f.write(f' epoch = {epoch + 1} \n')
                for i in range(len(confusion_matrix)):
                    for j in range(len(confusion_matrix[i])):
                        f.write(' ' + str(confusion_matrix[i][j]) + ' ')
                    f.write('\n')
                f.write('\n')

            score_save_dir = os.path.join(args.output_dir, 'score.txt')
            with open(score_save_dir, 'a') as f:
                dices = []
                f.write(f' epoch = {epoch + 1} | ')
                for k in score_dict:
                    dice = float(score_dict[k])
                    f.write(f'class {k} = {dice} ')
                    dices.append(dice)
                dice_coeff = sum(dices) / len(dices)
                f.write(f'| dice_coeff = {dice_coeff}')
                f.write(f'\n')
    accelerator.end_training()

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
    parser.add_argument("--use_layer_norm", action='store_true')
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
    args = parser.parse_args()
    passing_argument(args)
    from data.dataset_multi import passing_mvtec_argument
    passing_mvtec_argument(args)
    main(args)