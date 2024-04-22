import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
import torch
from torch import nn
import os
from attention_store import AttentionStore
from data import call_dataset
from diffusers import DDPMScheduler
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
from monai.utils import DiceCEReduction, LossReduction
from monai.losses import FocalLoss
from monai.losses import DiceLoss, DiceCELoss
from torch.nn import functional as F

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

    print(f'\n step 3. preparing accelerator')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f'\n step 4. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    condition_model, vae, unet, network = call_model_package(args, weight_dtype, accelerator)
    segmentation_head = SemanticModel(n_classes=args.n_classes,
                                      mask_res=args.mask_res,
                                      use_layer_norm = args.use_layer_norm,
                                      use_instance_norm = args.use_instance_norm,)

    reduction_net = None
    if args.reducing_redundancy :

        # i think it is not that good ... (is there any other way to reduce redundancy ?)
        # image info is much redundancy than text
        # so, i think it is better to reduce redundancy in image info

        class ReductionNet(nn.Module):
            def __init__(self, cross_dim, class_num):

                super(ReductionNet, self).__init__()
                self.layer = nn.Sequential(nn.Linear(cross_dim, class_num),
                                               nn.Softmax(dim=-1))

            def forward(self, x):
                class_embedding = x[:, 0, :]
                org_x = x[:, 1:, :]  # x = [1,196,768]

                reduct_x = self.layer(org_x)  # x = [1,196,3]
                if args.use_weighted_reduct :
                    weight_x = reduct_x.permute(0, 2, 1).contiguous()  # x = [1,3,196]
                    weight_scale = torch.sum(weight_x, dim=-1)
                    reduct_x = torch.matmul(weight_x, org_x)  # x = [1,3,768]
                    # normalizing in channel dimention ***
                    #reduct_x = F.normalize(reduct_x, p=2, dim=-1)
                    reduct_x = reduct_x / weight_scale.unsqueeze(-1)
                else :
                    reduct_x = reduct_x.permute(0, 2, 1).contiguous()    # x = [1,3,196]
                    reduct_x = torch.matmul(reduct_x, org_x) # [1,3,196] [1,196,768] = [1,3,768]
                    reduct_x = F.normalize(reduct_x, p=2, dim=-1)

                if class_embedding.dim() != 3:
                    class_embedding = class_embedding.unsqueeze(0)
                if class_embedding.dim() != 3:
                    class_embedding = class_embedding.unsqueeze(0)

                x = torch.cat([class_embedding, reduct_x], dim=1)

                return x

        reduction_net = ReductionNet(768, args.n_classes)

    print(f'\n step 4. dataset and dataloader')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader, test_dataloader, tokenizer = call_dataset(args)

    print(f'\n step 5. optimizer')
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr,
                                                        args.unet_lr,
                                                        args.learning_rate) # all trainable params
    if args.reducing_redundancy :
        trainable_params.append({"params": reduction_net.parameters(), "lr": args.learning_rate})
    trainable_params.append({"params": segmentation_head.parameters(),
                             "lr": args.learning_rate})
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)

    print(f'\n step 6. lr')
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 7. loss function')
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
    condition_model = accelerator.prepare(condition_model)
    condition_models = transform_models_if_DDP([condition_model])
    segmentation_head, unet, network, optimizer, train_dataloader, test_dataloader, lr_scheduler = \
      accelerator.prepare(segmentation_head, unet, network, optimizer, train_dataloader, test_dataloader, lr_scheduler)
    if args.reducing_redundancy :
        reduction_net = accelerator.prepare(reduction_net)
        reduction_net = transform_models_if_DDP([reduction_net])[0]
    unet, network = transform_models_if_DDP([unet, network])
    segmentation_head = transform_models_if_DDP([segmentation_head])[0]
    if args.gradient_checkpointing:
        unet.train()
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
    kl_weight = 1e-6
    for epoch in range(args.start_epoch, args.max_train_epochs):

        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            device = accelerator.device
            loss_dict = {}

            encoder_hidden_states = None  # torch.tensor((1,1,768)).to(device)

            if not args.without_condition:
                if args.use_image_condition:
                    if not args.image_model_training:
                        with torch.no_grad():
                            output, pix_embedding = condition_model(**batch["image_condition"])
                            encoder_hidden_states = output.last_hidden_state  # [batch, 197, 768]
                            if args.not_use_cls_token:
                                encoder_hidden_states = encoder_hidden_states[:, 1:, :]
                            if args.only_use_cls_token:
                                encoder_hidden_states = encoder_hidden_states[:, 0, :]
                            if args.reducing_redundancy:
                                encoder_hidden_states = reduction_net(encoder_hidden_states)
                    else:
                        with torch.set_grad_enabled(True):
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
                if encoder_hidden_states is not None:
                    if encoder_hidden_states.dim() != 3:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                    if encoder_hidden_states.dim() != 3:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)

                noise_pred = unet(latents, 0, encoder_hidden_states,
                                      trg_layer_list=args.trg_layer_list).sample

            target = torch.randn_like(noise_pred)
            noise_loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none").mean(
                [1, 2, 3])

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
            # ------------------------------------------------------------------------------------------------------------
            # [1] generator loss
            # ------------------------------------------------------------------------------------------------------------
            # [2] origin loss
            masks_pred_ = masks_pred.permute(0, 2, 3, 1).contiguous().view(-1, masks_pred.shape[-1]).contiguous()
            if args.use_dice_ce_loss:
                loss = loss_dicece(input=masks_pred, target=batch['gt'].to(dtype=weight_dtype))
            else:  # [5.1] Multiclassification Loss
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

            if args.use_noise_pred_loss:
                loss_dict['noise_loss'] = noise_loss.mean().item()
                loss = loss + noise_loss.mean()

            loss = loss.mean()
            current_loss = loss.detach().item()

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
            if args.reducing_redundancy :
                save_model(args,
                           saving_folder='reduction_net',
                           saving_name=f'reduction-{saving_epoch}.pt',
                           unwrapped_nw=accelerator.unwrap_model(reduction_net),
                           save_dtype=save_dtype)

        # ----------------------------------------------------------------------------------------------------------- #
        # [7] evaluate
        loader = test_dataloader
        if args.check_training:
            print(f'test with training data')
            loader = train_dataloader

        score_dict, confusion_matrix, _ = evaluation_check(segmentation_head, loader, accelerator.device,
                                                           condition_model, unet, vae, controller, weight_dtype, epoch,
                                                           reduction_net, None, args)

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
    args = parser.parse_args()
    passing_argument(args)
    from data.dataset_multi import passing_mvtec_argument
    passing_mvtec_argument(args)
    main(args)