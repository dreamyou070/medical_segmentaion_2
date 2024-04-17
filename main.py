from model.blip import blip_decoder
import argparse, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
from torch import nn
from attention_store import AttentionStore
from data import call_dataset
from model.segmentation_unet import SemanticSeg_Gen
from model.diffusion_model import transform_models_if_DDP
from utils.evaluate_3 import evaluation_check
from model.unet import unet_passing_argument
from utils import prepare_dtype, arg_as_list
from utils.attention_control import passing_argument, register_attention_control
from utils.accelerator_utils import prepare_accelerator
from utils.optimizer import get_optimizer, get_scheduler_fix
from utils.loss import FocalLoss, Multiclass_FocalLoss
from monai.utils import LossReduction
from torch.nn import L1Loss
from monai.losses import FocalLoss
from monai.losses import DiceLoss, DiceCELoss
from utils.losses import PatchAdversarialLoss
from model.lora_2 import create_network_2
from model.diffusion_model import load_target_model
import os
import torch
from utils.saving import save_model
from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads_3D_4D, reshape_batch_dim_to_heads_3D_3D
from model.pe import AllPositionalEmbedding
from safetensors.torch import load_file

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

    print(f' step 3. load model')
    weight_dtype, save_dtype = prepare_dtype(args)

    print(f' (3.2) load stable diffusion model')
    # [1] diffusion
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)

    text_encoder.requires_grad_(False)
    text_encoder.to(dtype=weight_dtype)

    vae.requires_grad_(False)
    vae.to(dtype=weight_dtype)
    vae.eval()

    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)

    # [2] pe
    position_embedder = None
    position_embedder = AllPositionalEmbedding(pe_do_concat=args.pe_do_concat,
                                               do_semantic_position=args.do_semantic_position, )

    if args.position_embedder_weights is not None:
        position_embedder_state_dict = load_file(args.position_embedder_weights)
        position_embedder.load_state_dict(position_embedder_state_dict)
        position_embedder.to(dtype=weight_dtype)


    # [2] blip model
    image_size = 384
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
    blip_model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    blip_image_model, blip_text_model = blip_model.visual_encoder, blip_model.text_decoder

    # [3] lora network
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value
    network = create_network_2(1.0,
                               args.network_dim,
                               args.network_alpha,
                               vae,
                               image_condition = blip_image_model,
                               text_condition = blip_text_model,
                               unet = unet,
                               neuron_dropout=args.network_dropout,
                               **net_kwargs, )
    network.apply_to(blip_text_model, blip_image_model, unet,
                     apply_image_encoder=True,
                     apply_text_encoder=False,
                     apply_unet=True)

    unet = unet.to(accelerator.device, dtype=weight_dtype)
    unet.eval()

    blip_text_model = blip_text_model.to(accelerator.device, dtype=weight_dtype)
    blip_text_model.eval()

    vae = vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()

    blip_image_model = blip_image_model.to(accelerator.device, dtype=weight_dtype)
    blip_image_model.eval()

    if args.network_weights is not None:
        print(f' * loading network weights')
        info = network.load_weights(args.network_weights)
    network.to(weight_dtype)
    segmentation_head = SemanticSeg_Gen(n_classes=args.n_classes,
                                        mask_res=args.mask_res,
                                        high_latent_feature=args.high_latent_feature,
                                        init_latent_p=args.init_latent_p)

    print(f'\n step 4. dataset and dataloader')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader, test_dataloader, tokenizer = call_dataset(args)

    print(f'\n step 5. optimizer')
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    trainable_params = network.prepare_optimizer_params(args.image_encoder_lr,
                                                        args.text_encoder_lr,
                                                        args.unet_lr,
                                                        args.learning_rate)  # all trainable params
    blip_trainable_params = trainable_params[1]

    simple_linear = nn.Linear(576, 3)

    trainable_params.append({"params": segmentation_head.parameters(), "lr": args.learning_rate})
    trainable_params.append({"params": simple_linear.parameters(), "lr": args.learning_rate})
    if args.use_position_embedder:
        trainable_params.append({"params": position_embedder.parameters(), "lr": args.learning_rate})
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)

    print(f'\n step 6. lr')
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 7. loss function')
    l1_loss = L1Loss()
    l2_loss = nn.MSELoss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_CE = nn.CrossEntropyLoss()
    loss_FC = Multiclass_FocalLoss()
    if args.use_monai_focal_loss:
        loss_FC = FocalLoss(include_background=False,
                            to_onehot_y=True,
                            gamma=2.0,
                            weight=None,
                            reduction=LossReduction.MEAN,
                            use_softmax=True)
    loss_Dice = DiceLoss(include_background=False,
                         to_onehot_y=False,
                         sigmoid=False,
                         softmax=True,
                         other_act=None,
                         squared_pred=False,
                         jaccard=False,
                         reduction=LossReduction.MEAN,
                         smooth_nr=1e-5,
                         smooth_dr=1e-5,
                         batch=False,
                         weight=None)

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
    blip_model.visual_encoder, blip_model.text_decoder = blip_image_model, blip_text_model
    #blip_image_model = accelerator.prepare(blip_image_model)
    #blip_image_models = transform_models_if_DDP([blip_image_model])
    #blip_text_model = accelerator.prepare(blip_text_model)
    #blip_text_models = transform_models_if_DDP([blip_text_model])

    blip_model = accelerator.prepare(blip_model)
    blip_model = transform_models_if_DDP([blip_model])[0]

    segmentation_head, unet, network, optimizer, train_dataloader, test_dataloader, lr_scheduler = \
        accelerator.prepare(segmentation_head, unet, network, optimizer, train_dataloader, test_dataloader,
                            lr_scheduler)
    if args.use_position_embedder:
        position_embedder = accelerator.prepare(position_embedder)

    simple_linear = accelerator.prepare(simple_linear)
    unet, network = transform_models_if_DDP([unet, network])
    segmentation_head = transform_models_if_DDP([segmentation_head])[0]
    position_embedder = accelerator.prepare(position_embedder)[0]
    if args.gradient_checkpointing:
        unet.train()
        segmentation_head.train()
        blip_model.train()
        position_embedder.train()
        """
        for i_enc in blip_image_models:
            i_enc.train()
            if args.train_image_encoder:
                i_enc.visual_model.embeddings.requires_grad_(True)

        for t_enc in blip_text_models:
            t_enc.train()
            if args.train_text_encoder:
                t_enc.text_model.embeddings.requires_grad_(True)
        """
    else:
        unet.eval()
        """
        for t_enc in blip_text_models:
            t_enc.eval()
        del t_enc
        for i_enc in blip_image_models:
            i_enc.eval()
        del i_enc
        network.prepare_grad_etc()
        """

    print(f'\n step 9. registering saving tensor')
    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f'\n step 10. Training !')
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []

    for epoch in range(args.start_epoch, args.max_train_epochs):

        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")
        for step, batch in enumerate(train_dataloader):
            device = accelerator.device
            loss_dict = {}
            # -----------------------------------------------------------------------------------------------------------------------------
            # [1] lm_loss
            caption = batch['caption']       # ['this picture is of b n']
            image = batch['image_condition'] # [batch, 3, 384, 384]

            # why lm_loss does not reducing ??
            lm_loss, image_feature = blip_model(image, caption) # [batch, 577, 768]

            cls_token = image_feature[:, 0, :]
            image_features = image_feature[:, 1:, :]
            image_feature_transpose = image_features.transpose(1, 2)  # [batch, dim, pixels]


            image_feat = simple_linear(image_feature_transpose).transpose(1, 2)  # [batch, pixels, dim]
            encoder_hidden_states = torch.cat((cls_token.unsqueeze(1), image_feat), dim=1)
            #print(condition.shape)  # torch.Size([1, 4, 768])

            """
            # -----------------------------------------------------------------------------------------------------------------------------
            if args.use_image_condition:
                with torch.set_grad_enabled(True):
                    encoder_hidden_states = image_feature
            if args.use_text_condition:
                with torch.set_grad_enabled(True):
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]  # [batch, 77, 768]
            """
            image = batch['image'].to(dtype=weight_dtype)  # 1,3,512,512
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)  # 1,128*128
            gt = batch['gt'].to(dtype=weight_dtype)  # 1,3,256,256
            gt = gt.permute(0, 2, 3, 1).contiguous()  # .view(-1, gt.shape[-1]).contiguous()   # 1,256,256,3
            gt = gt.view(-1, gt.shape[-1]).contiguous()
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
            with torch.set_grad_enabled(True):
                unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                     noise_type=position_embedder)
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
            masks_pred = segmentation_head(x16_out, x32_out, x64_out, latents)
            # [2] origin loss
            masks_pred_ = masks_pred.permute(0, 2, 3, 1).contiguous().view(-1, masks_pred.shape[-1]).contiguous()
            if args.use_dice_ce_loss:
                loss = loss_dicece(input=masks_pred, target=batch['gt'].to(dtype=weight_dtype))
            else:
                # [5.1] Multiclassification Loss
                loss = loss_CE(masks_pred_, gt_flat.squeeze().to(torch.long))  # 128*128
                loss_dict['cross_entropy_loss'] = loss.item()
                # [5.2] Focal Loss
                focal_loss = loss_FC(masks_pred_, gt_flat.squeeze().to(masks_pred.device))  # N
                if args.use_monai_focal_loss: focal_loss = focal_loss.mean()
                loss += focal_loss
                loss_dict['focal_loss'] = focal_loss.item()
                # [5.3] Dice Loss
                if args.use_dice_loss:
                    dice_loss = loss_Dice(masks_pred, gt)
                    loss += dice_loss
                    loss_dict['dice_loss'] = dice_loss.item()
                loss = loss.mean()
            loss = loss * args.segmentation_loss_weight
            loss = loss.mean()
            loss_dict['lm_loss'] = lm_loss.item()

            # -----------------------------------------------------------------------------------------------------------------------------
            total_loss = loss # + lm_loss
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
            save_model(args,
                       saving_folder='segmentation',
                       saving_name=f'segmentation-{saving_epoch}.pt',
                       unwrapped_nw=accelerator.unwrap_model(segmentation_head),
                       save_dtype=save_dtype)

        # ----------------------------------------------------------------------------------------------------------- #

        # [7] evaluate
        loader = test_dataloader
        if args.check_training:
            print(f'test with training data')
            loader = train_dataloader
        score_dict, confusion_matrix, _ = evaluation_check(segmentation_head, loader, accelerator.device,
                                                           blip_model, unet, vae, controller, weight_dtype, epoch,
                                                           simple_linear, position_embedder, args,)
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
    parser.add_argument("--network_alpha", type=float, default=4, help="alpha for LoRA weight scaling, default 1 ", )
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
    parser.add_argument('--image_encoder_lr', type=float, default=1e-5)
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
    parser.add_argument("--use_base_prompt", action='store_true')
    parser.add_argument("--use_key_word", action='store_true')
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    from data.dataset_multi import passing_mvtec_argument

    passing_mvtec_argument(args)
    main(args)


