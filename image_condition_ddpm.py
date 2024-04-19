import argparse, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
from torch import nn
from model.diffusion_model import transform_models_if_DDP
from model.unet import unet_passing_argument
from utils.attention_control import passing_argument
import time
from utils.accelerator_utils import prepare_accelerator
from diffusers import DDPMScheduler
import os
import torch
from utils import prepare_dtype, arg_as_list
from data.image_conditioned_generating_dataset import call_dataset
from gen.generative.networks.nets import DiffusionModelUNet
from gen.generative.inferers import DiffusionInferer
from transformers import ViTModel
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
class simple_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_projector = nn.Sequential(nn.Linear(768, 512),
                                          nn.GELU(),
                                          nn.Linear(512, 768), )

    def forward(self, x):
        return self.mm_projector(x)

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
    print(f' (3.1) unet model')

    model = DiffusionModelUNet(spatial_dims=2,
                               in_channels=3,
                               out_channels=3,
                               num_channels=(128, 256, 256),
                               attention_levels=(False, True, True),
                               num_res_blocks=1,
                               cross_attention_dim=768,
                               num_head_channels=256,
                               with_conditioning=True )


    print(f' (3.2) condition model')
    condition_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    condition_model = condition_model.to(accelerator.device, dtype=weight_dtype)
    condition_model.eval()

    simple_linear = simple_net()
    simple_linear.to(dtype=weight_dtype)

    print(f' (3.3) scheduler')
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    print(f'\n step 4. dataset and dataloader')
    if args.seed is None : args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader, test_dataloader = call_dataset(args)

    print(f'\n step 5. optimizer') # oh.. i did not put simple linear in optimizer
    trainable_params = [{'params': model.parameters(),
                         'lr': args.learning_rate,}]
    trainable_params += [{'params': simple_linear.parameters(), 'lr': args.learning_rate}]
    optimizer = torch.optim.Adam(trainable_params)
    inferer = DiffusionInferer(scheduler)

    print(f'\n step 7. loss function')
    l2_loss = nn.MSELoss(reduction='none')

    print(f'\n step 8. model to device')
    optimizer = accelerator.prepare(optimizer)
    train_dataloader, test_dataloader = accelerator.prepare(train_dataloader, test_dataloader)
    model,simple_linear = accelerator.prepare(model,simple_linear)
    simple_linear,condition_model, model = transform_models_if_DDP([simple_linear,condition_model, model])

    print(f'\n step 10. Training !')
    progress_bar = tqdm(range(args.max_train_epochs), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []
    noise_scheduler = DDPMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    num_train_timesteps=1000,
                                    clip_sample=False)
    val_interval = 5
    epoch_loss_list = []
    val_epoch_loss_list = []
    scaler = GradScaler()
    total_start = time.time()
    device = accelerator.device

    for epoch in range(args.max_train_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        """
        for step, batch in progress_bar:

            optimizer.zero_grad(set_to_none=True)

            # final output
            images = batch["image"].to(dtype=weight_dtype) # mask image -> [3,256,256]

            # [2] condition image (dtype = float32, 1,3,224,224)
            condition_pixel = batch['condition_image']['pixel_values'].to(dtype=weight_dtype, )
            batch['condition_image']['pixel_values'] = condition_pixel
            feat = condition_model(**batch['condition_image']).last_hidden_state  # processor output
            encoder_hidden_states = simple_linear(feat.contiguous())  # [batch=1, 197, 768]
            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn_like(images).to(device)
                # Create timesteps
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()

                # Get model prediction
                noise_pred = inferer(inputs=images,
                                     diffusion_model=model,
                                     noise=noise,
                                     timesteps=timesteps,
                                     condition = encoder_hidden_states)
                loss = F.mse_loss(noise_pred.float(),
                                  noise.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))
        """
        # --------------------------------------------------------- Validation --------------------------------------------------------- #
        accelerator.wait_for_everyone()
        model.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(test_dataloader):
            if step == 1 :
                images = batch["image"].to(dtype=weight_dtype)  # [2
                # [2] condition image (dtype = float32, 1,3,224,224)
                condition_pixel = batch['condition_image']['pixel_values'].to(dtype=weight_dtype, )
                batch['condition_image']['pixel_values'] = condition_pixel
                feat = condition_model(**batch['condition_image']).last_hidden_state  # processor output
                encoder_hidden_states = simple_linear(feat.contiguous())  # [batch=1, 197, 768]
                """
                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(device)
                        timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
                        noise_pred = inferer(inputs=images,
                                             diffusion_model=model, noise=noise, timesteps=timesteps,
                                             condition = encoder_hidden_states)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
                """
            else :
                break
        # recording val loss
        #val_loss_text = os.path.join(record_save_dir, 'val_loss.txt')
        #with open(val_loss_text, 'a') as f:
        #    f.write(f' epoch = {epoch}, val_loss = {val_epoch_loss / (step + 1)} \n')



        # --------------------------------------------------------- Validation --------------------------------------------------------- #
        if is_main_process:
            # Sampling image during training
            noise = torch.randn((1,3,256,256)).to(accelerator.device, dtype=weight_dtype)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(enabled=True):
                image = inferer.sample(input_noise=noise,
                                       diffusion_model=model,
                                       scheduler=scheduler,
                                       conditioning=encoder_hidden_states,
                                       mode = "crossattn") # tensor image (1,3,256,256)
                print(f'image = {image.shape}')
                # tensor to pil image
                from matplotlib import pyplot as plt
                plt.figure(figsize=(2, 2))
                plt.imshow(image[0, 0].cpu())
                plt.tight_layout()
                plt.axis("off")
                sample_dir = os.path.join(output_dir, 'sample')
                os.makedirs(sample_dir, exist_ok=True)
                plt.savefig(os.path.join(sample_dir, f"sample_{epoch}.png"))

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")




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
    parser.add_argument("--sample_sampler", type=str, default="ddim",
                        choices=["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver",
                                 "dpmsolver++",
                                 "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", ],
                        help=f"sampler (scheduler) type for sample images / サンプル出力時のサンプラー（スケジューラ）の種類", )
    parser.add_argument("--v_parameterization", action="store_true",
                        help="enable v-parameterization training / v-parameterization学習を有効にする"
                        )
    parser.add_argument("--num_inference_steps", type=int, default=30, )
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    from data.dataset_multi import passing_mvtec_argument

    passing_mvtec_argument(args)
    from data.image_conditioned_generating_dataset import passing_mvtec_argument as passing_data

    passing_data(args)
    main(args)