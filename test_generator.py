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
from monai.utils import DiceCEReduction, LossReduction
from monai.losses import FocalLoss
from monai.losses import DiceLoss, DiceCELoss
from model.reduction_model import ReductionNet
from model.vision_condition_head import vision_condition_head
from ignite.metrics.confusion_matrix import ConfusionMatrix
import torch
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
import torch.nn.functional as F
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
    segmentation_head = SemanticModel(n_classes=args.n_classes,
                                      mask_res=args.mask_res,
                                      use_layer_norm = args.use_layer_norm)
    reduction_net = None
    if args.reducing_redundancy :
        reduction_net = ReductionNet(768, args.n_classes,
                                     use_weighted_reduct=args.use_weighted_reduct)
    vision_head = None
    if args.image_processor == 'pvt' :
        vision_head = vision_condition_head(reverse = args.reverse)

    position_embedder = None
    if args.use_position_embedder :
        from model.pe import AllPositionalEmbedding
        position_embedder = AllPositionalEmbedding()
        if args.position_embedder_weights is not None:
            position_embedder.load_state_dict(torch.load(args.position_embedder_weights))

    class AnomalFeatureGenerator(torch.nn.Module):
        """ anomal feature mapping class """
        def __init__(self):
            super(AnomalFeatureGenerator, self).__init__()
            self.layer1 = torch.nn.Linear(160, 320)
            self.layer2 = torch.nn.Linear(320, 160)

        def forward(self, random_feature):
            random_feature = self.layer1(random_feature)
            random_feature = self.layer2(random_feature)
            return random_feature

    anomal_generator = AnomalFeatureGenerator()
    if args.anomal_generator_weights is not None:
        anomal_generator.load_state_dict(torch.load(args.anomal_generator_weights))

    print(f'\n step 8. model to device')
    condition_model = accelerator.prepare(condition_model)
    condition_models = transform_models_if_DDP([condition_model])
    segmentation_head, unet, network = accelerator.prepare(segmentation_head, unet, network)
    anomal_generator = accelerator.prepare(anomal_generator)
    anomal_generator = transform_models_if_DDP([anomal_generator])[0]
    if args.reducing_redundancy :
        reduction_net = accelerator.prepare(reduction_net)
        reduction_net = transform_models_if_DDP([reduction_net])[0]
    if args.use_position_embedder :
        position_embedder = accelerator.prepare(position_embedder)
        position_embedder = transform_models_if_DDP([position_embedder])[0]
    if args.image_processor == 'pvt' :
        vision_head = accelerator.prepare(vision_head)
        vision_head = transform_models_if_DDP([vision_head])[0]

    unet, network = transform_models_if_DDP([unet, network])
    segmentation_head = transform_models_if_DDP([segmentation_head])[0]

    print(f'\n step 9. loss function')
    loss_dicece = DiceCELoss(include_background=False,
                             to_onehot_y=False,
                             sigmoid=False,
                             softmax=True,
                             squared_pred=True,
                             lambda_dice=args.dice_weight,
                             smooth_nr=1e-5,
                             smooth_dr=1e-5,
                             weight=None, )

    print(f'\n step 10. Training !')
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []
    kl_weight = 1e-6
    device = accelerator.device

    y_true_list, y_pred_list = [], []

    for epoch in range(args.start_epoch, args.max_train_epochs):

        # [1] generator pseudo sample
        random_feature = torch.randn((args.mask_res*args.mask_res, 160)).to(dtype=weight_dtype, device = device)
        pseudo_sample = anomal_generator(random_feature).permute(1,0).contiguous()
        dim = pseudo_sample.shape[0]
        pseudo_feature = pseudo_sample.view(dim, args.mask_res, args.mask_res).contiguous().unsqueeze(0) # 1, 160, 265,265

        # [2] pseudo answer
        pseudo_label = torch.ones((1, args.n_classes, args.mask_res, args.mask_res)).to(dtype=weight_dtype, device=device)
        pseudo_label[:, 0, :, :] = 0 # all class 1 samples
        gt_flat = torch.ones(args.mask_res*args.mask_res).to(dtype=weight_dtype, device=device)
        pseudo_masks_pred = segmentation_head.segment_feature(pseudo_feature)  # 1,2,265,265
        pseudo_loss = loss_dicece(input=pseudo_masks_pred,   # [class, 256,256]
                                  target=pseudo_label.to(dtype=weight_dtype,
                                                         device=accelerator.device))  # [class, 256,256]
        # pseudo loss make
        class_num = args.n_classes
        mask_pred_argmax = torch.argmax(pseudo_masks_pred, dim=1).flatten()  # 256*256
        y_pred_list.append(mask_pred_argmax)
        y_true = gt_flat.squeeze()
        y_true_list.append(y_true)

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
    # showing
    print(f'\n confusion matrix : {confusion_matrix}')
    print(f'\n IOU : {IOU_dict}')


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
    parser.add_argument("--reverse", action='store_true')
    parser.add_argument("--online_pseudo_loss", action='store_true')
    parser.add_argument("--only_online_pseudo_loss", action='store_true')
    parser.add_argument("--pseudo_loss_weight", type=float, default=1)
    parser.add_argument("--anomal_loss_weight", type=float, default=1)
    parser.add_argument("--anomal_mse_loss", action='store_true')
    parser.add_argument("--anomal_kl_loss", action='store_true')
    parser.add_argument("--anomal_generator_weights", type=str)

    args = parser.parse_args()
    passing_argument(args)
    from data.dataset_multi import passing_mvtec_argument
    passing_mvtec_argument(args)
    main(args)