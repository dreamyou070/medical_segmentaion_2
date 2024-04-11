import os
import argparse
from model.lora import LoRANetwork,LoRAInfModule
from attention_store import AttentionStore
from utils.attention_control import passing_argument
from model.unet import unet_passing_argument
from utils.attention_control import register_attention_control
from accelerate import Accelerator
from model.tokenizer import load_tokenizer
from utils import prepare_dtype
from model.diffusion_model import load_target_model
from model.pe import AllPositionalEmbedding
from safetensors.torch import load_file
# pip install pytorch-ignite
import torch
from accelerate.utils import set_seed
import random
from data import call_dataset
from utils.evaluate import evaluation_check


def reshape_batch_dim_to_heads(tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = 8
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len,
                            dim)  # 1,8,pix_num, dim -> 1,pix_nun, 8,dim
    tensor = tensor.permute(0, 2, 1, 3).contiguous().reshape(batch_size // head_size, seq_len,
                                                             dim * head_size)  # 1, pix_num, long_dim
    res = int(seq_len ** 0.5)
    tensor = tensor.view(batch_size // head_size, res, res, dim * head_size).contiguous()
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # 1, dim, res,res
    return tensor

def main(args):

    print(f'\n step 1. accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with,
                              project_dir='log')

    print(f'\n step 2. test data loader')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader, test_dataloader = call_dataset(args)
    loader = test_dataloader
    if args.do_train_check:
        loader = train_dataloader

    print(f'\n step 3. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
    print(f' (3.1) position embedder')
    position_embedder = None
    if args.use_position_embedder:
        position_embedder = AllPositionalEmbedding()
    print(f' (3.2) segmentation head')
    if args.use_original_seg_unet:
        from sub.segmentation_unet import Segmentation_Head_a, Segmentation_Head_b, Segmentation_Head_c
        segmentation_head_class = Segmentation_Head_a
        if args.aggregation_model_b:
            segmentation_head_class = Segmentation_Head_b
        if args.aggregation_model_c:
            segmentation_head_class = Segmentation_Head_c
        segmentation_head = segmentation_head_class(n_classes=args.n_classes,
                                                    mask_res=args.mask_res)
    elif args.use_new_seg_unet:
        from model.segmentation_unet import Segmentation_Head_a, Segmentation_Head_b, Segmentation_Head_c
        segmentation_head_class = Segmentation_Head_a
        if args.aggregation_model_b:
            segmentation_head_class = Segmentation_Head_b
        if args.aggregation_model_c:
            segmentation_head_class = Segmentation_Head_c
        segmentation_head = segmentation_head_class(n_classes=args.n_classes,
                                                    mask_res=args.mask_res,
                                                    norm_type=args.norm_type,
                                                    non_linearity=args.non_linearity, )
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f'\n step 4. inference')
    models = os.listdir(args.network_folder)
    network = LoRANetwork(text_encoder=text_encoder,
                          unet=unet,
                          lora_dim=args.network_dim,
                          alpha=args.network_alpha,
                          module_class=LoRAInfModule)
    network.apply_to(text_encoder, unet, True, True)
    raw_state_dict = network.state_dict()
    raw_state_dict_orig = raw_state_dict.copy()

    for model in models:
        network_model_dir = os.path.join(args.network_folder, model)
        lora_name, ext = os.path.splitext(model)
        lora_epoch = int(lora_name.split('-')[-1])
        lora_epoch = str(lora_epoch).zfill(6)

        # [2] load network
        anomal_detecting_state_dict = load_file(network_model_dir)
        for k in anomal_detecting_state_dict.keys():
            raw_state_dict[k] = anomal_detecting_state_dict[k]
        network.load_state_dict(raw_state_dict)
        network.to(accelerator.device, dtype=weight_dtype)

        # [3] segmentation model
        parent = os.path.split(args.network_folder)[0]
        seg_base_dir = os.path.join(parent, f'segmentation')
        pretrained_seg_dir = os.path.join(seg_base_dir, f'segmentation-{lora_epoch}.pt')
        segmentation_head.load_state_dict(torch.load(pretrained_seg_dir))
        segmentation_head.to(accelerator.device, dtype=weight_dtype)

        # [1] loead pe
        if args.use_position_embedder:
            pe_base_dir = os.path.join(parent, f'position_embedder')
            position_embedder_state_dict = torch.load(os.path.join(pe_base_dir, f'position_embedder-{lora_epoch}.pt'))
            position_embedder.load_state_dict(position_embedder_state_dict)
            position_embedder.to(accelerator.device, dtype=weight_dtype)

        # [5] inference
        score_dict, confusion_matrix, dice_coeff = evaluation_check(segmentation_head, loader, accelerator.device,
                                                                    text_encoder, unet, vae, controller, weight_dtype,
                                                                    position_embedder, args)
        # [6] saving
        test_save_dir = os.path.join(args.output_dir, f'test')
        os.makedirs(test_save_dir, exist_ok=True)
        # saving
        print(f'  - precision dictionary = {score_dict}')
        print(f'  - confusion_matrix = {confusion_matrix}')
        confusion_matrix = confusion_matrix.tolist()
        confusion_save_dir = os.path.join(test_save_dir, 'confusion.txt')
        with open(confusion_save_dir, 'a') as f:
            f.write(f' epoch = {lora_epoch + 1} \n')
            for i in range(len(confusion_matrix)):
                for j in range(len(confusion_matrix[i])):
                    f.write(' ' + str(confusion_matrix[i][j]) + ' ')
                f.write('\n')
            f.write('\n')

        score_save_dir = os.path.join(args.output_dir, 'score.txt')
        with open(score_save_dir, 'a') as f:
            dices = []
            f.write(f' epoch = {lora_epoch + 1} | ')
            for k in score_dict:
                dice = float(score_dict[k])
                f.write(f'class {k} = {dice} ')
                dices.append(dice)
            dice_coeff = sum(dices) / len(dices)
            f.write(f'| dice_coeff = {dice_coeff}')
            f.write(f'\n')

        # [7] be original
        network.load_state_dict(raw_state_dict_orig)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument("--resize_shape", type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int, default=64)
    parser.add_argument('--network_alpha', type=float, default=4)
    parser.add_argument('--network_folder', type=str)
    parser.add_argument("--lowram", action="store_true", )
    # step 4. dataset and dataloader
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument('--data_path', type=str,
                        default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bagel')
    # step 6
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--prompt", type=str, default="bagel", )
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--latent_res", type=int, default=64)
    parser.add_argument("--single_layer", action='store_true')
    parser.add_argument("--use_noise_scheduler", action='store_true')
    parser.add_argument('--min_timestep', type=int, default=0)
    parser.add_argument('--max_timestep', type=int, default=500)
    # step 8. test
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--threds", type=arg_as_list,default=[0.85,])
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--do_normalized_score", action='store_true')
    parser.add_argument("--d_dim", default=320, type=int)
    parser.add_argument("--thred", default=0.5, type=float)
    parser.add_argument("--image_classification_layer", type=str)
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--vae_scale_factor", type=float, default=0.18215)
    parser.add_argument("--object_crop", action='store_true')
    parser.add_argument("--use_multi_position_embedder", action='store_true')
    parser.add_argument("--all_positional_embedder", action='store_true')
    parser.add_argument("--all_positional_self_cross_embedder", action='store_true')
    parser.add_argument("--patch_positional_self_embedder", action='store_true')
    parser.add_argument("--all_self_cross_positional_embedder", action='store_true')
    parser.add_argument("--use_global_conv", action='store_true')
    parser.add_argument("--do_train_check", action='store_true')
    parser.add_argument("--vae_pretrained_dir", type=str)
    parser.add_argument("--use_global_network", action='store_true')
    parser.add_argument("--text_truncate", action='store_true')
    parser.add_argument("--test_with_xray", action='store_true')
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--use_batchnorm", action='store_true')
    parser.add_argument("--use_original_seg_unet", action='store_true')
    parser.add_argument("--use_new_seg_unet", action='store_true')
    parser.add_argument("--aggregation_model_b", action='store_true')
    parser.add_argument("--aggregation_model_c", action='store_true')
    parser.add_argument("--norm_type", type=str, default='batch')
    parser.add_argument("--non_linearity", type=str, default='relu')
    parser.add_argument("--mask_res", type=int, default=128)
    parser.add_argument("--image_folder_name", type=str, default='image_256')
    parser.add_argument("--gt_folder_name", type=str, default='mask_256')
    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    main(args)