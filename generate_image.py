import os
import argparse, torch
from model.lora import LoRANetwork,LoRAInfModule
from attention_store import AttentionStore
from utils.attention_control import passing_argument
from model.unet import unet_passing_argument
from utils.attention_control import register_attention_control
from accelerate import Accelerator
from model.tokenizer import load_tokenizer
from utils import prepare_dtype
from utils.model_utils import get_input_ids
from PIL import Image
import numpy as np
from model.diffusion_model import load_target_model
from safetensors.torch import load_file
from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads_3D_4D
from model.pe import AllPositionalEmbedding
from model.segmentation_unet import SemanticSeg
from model.autodecoder import AutoencoderKL
from matplotlib import pyplot as plt
from diffusers.image_processor import VAEImageProcessor

vae_scale_factor = 0.18215
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

def main(args):

    print(f'\n step 1. accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with,
                              project_dir='log')

    print(f'\n step 2. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
    print(f' (2.2) position embedder')
    position_embedder = None
    if args.use_position_embedder:
        position_embedder = AllPositionalEmbedding(pe_do_concat=args.pe_do_concat,
                                                   do_semantic_position = args.do_semantic_position)
    print(f' (2.3) segmentation head')
    segmentation_head = SemanticSeg(n_classes=args.n_classes, mask_res=args.mask_res, )

    print(f' (2.4) decoder')
    decoder_model = AutoencoderKL(spatial_dims=2,
                                  in_channels=1,
                                  out_channels=3,
                                  num_res_blocks=(2, 2, 2, 2), num_channels=(32, 64, 64, 64),
                                  attention_levels=(False, False, True, True),
                                  latent_channels=4,
                                  norm_num_groups=32, norm_eps=1e-6,
                                  with_encoder_nonlocal_attn=True, with_decoder_nonlocal_attn=True,
                                  use_flash_attention=False,
                                  use_checkpointing=False,
                                  use_convtranspose=False)


    print(f'\n step 2. accelerator and device')
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 3. inference')
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
        # [1] load pe
        parent = os.path.split(args.network_folder)[0]
        if args.use_position_embedder:
            pe_base_dir = os.path.join(parent, f'position_embedder')
            pretrained_pe_dir = os.path.join(pe_base_dir, f'position_embedder-{lora_epoch}.pt')
            position_embedder_state_dict = torch.load(pretrained_pe_dir)
            position_embedder.load_state_dict(position_embedder_state_dict)
            position_embedder.to(accelerator.device, dtype=weight_dtype)

        # [2] load network
        anomal_detecting_state_dict = load_file(network_model_dir)
        for k in anomal_detecting_state_dict.keys():
            raw_state_dict[k] = anomal_detecting_state_dict[k]
        network.load_state_dict(raw_state_dict)
        network.to(accelerator.device, dtype=weight_dtype)

        # [3] segmentation model
        seg_base_dir = os.path.join(parent, f'segmentation')
        pretrained_seg_dir = os.path.join(seg_base_dir, f'segmentation-{lora_epoch}.pt')
        segmentation_head.load_state_dict(torch.load(pretrained_seg_dir))
        segmentation_head.to(accelerator.device, dtype=weight_dtype)

        # [3] decoder file
        decoder_base_dir = os.path.join(parent, f'decoder')
        pretrained_decoder_dir = os.path.join(decoder_base_dir, f'decoder-{lora_epoch}.pt')
        decoder_model.load_state_dict(torch.load(pretrained_decoder_dir))
        decoder_model.to(accelerator.device, dtype=weight_dtype)

        # [4] files
        parent, _ = os.path.split(args.network_folder)
        if args.do_train_check :
            recon_base_folder = os.path.join(parent, 'generating_with_train_data')
        else :
            recon_base_folder = os.path.join(parent, 'generating_with_test_data')

        os.makedirs(recon_base_folder, exist_ok=True)
        lora_base_folder = os.path.join(recon_base_folder, f'lora_epoch_{lora_epoch}')
        os.makedirs(lora_base_folder, exist_ok=True)

        # [5] collector
        controller = AttentionStore()
        register_attention_control(unet, controller)

        # [6] save directory
        check_base_folder = os.path.join(lora_base_folder, f'my_check')
        os.makedirs(check_base_folder, exist_ok=True)
        answer_base_folder = os.path.join(lora_base_folder, f'scoring/{args.obj_name}/test')
        os.makedirs(answer_base_folder, exist_ok=True)

        # [1] test path
        test_img_folder = args.data_path
        if args.do_train_check :
            test_img_folder = os.path.join(os.path.split(test_img_folder)[0], 'train')
        mask_res_folders = os.listdir(test_img_folder)

        for mask_res_folder in mask_res_folders:
            folder_res = int(mask_res_folder.split('_')[-1])
            answer_anomal_folder = os.path.join(answer_base_folder, mask_res_folder)
            os.makedirs(answer_anomal_folder, exist_ok=True)
            save_base_folder = os.path.join(check_base_folder, mask_res_folder)
            os.makedirs(save_base_folder, exist_ok=True)
            anomal_folder_dir = os.path.join(test_img_folder, mask_res_folder)
            rgb_folder = os.path.join(anomal_folder_dir, f'image_{folder_res}')
            gt_folder = os.path.join(anomal_folder_dir, f'mask_{folder_res}')
            rgb_imgs = os.listdir(rgb_folder)


            for rgb_img in rgb_imgs:
                name, ext = os.path.splitext(rgb_img)
                rgb_img_dir = os.path.join(rgb_folder, rgb_img)
                pil_img = Image.open(rgb_img_dir).convert('RGB')
                org_h, org_w = pil_img.size

                # [1] read object mask
                input_img = pil_img
                trg_h, trg_w = input_img.size
                if accelerator.is_main_process:
                    # [2] Stable Diffusion
                    with torch.no_grad():

                        # [2.1] latent
                        img = np.array(input_img.resize((512, 512))) # [512,512,3]
                        image = torch.from_numpy(img).float() / 127.5 - 1
                        image = image.permute(2, 0, 1).unsqueeze(0).to(vae.device, weight_dtype) # [1,3,512,512]
                        latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor

                        # [2.2] text encoder
                        args.prompt = 'a picture of a b, n, e, t'
                        input_ids, attention_mask = get_input_ids(tokenizer, args.prompt)
                        encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device))["last_hidden_state"]

                        # [2.3] Unet
                        unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list)
                        query_dict, key_dict = controller.query_dict, controller.key_dict
                        controller.reset()
                        q_dict = {}
                        for layer in args.trg_layer_list:
                            query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                            res = int(query.shape[1] ** 0.5)
                            reshaped_query = reshape_batch_dim_to_heads_3D_4D(query)  # 1, res, res, dim
                            q_dict[res] = reshaped_query
                        x16_out, x32_out, x64_out = q_dict[16], q_dict[32], q_dict[64]
                        hidden_latent, masks_pred = segmentation_head(x16_out, x32_out, x64_out)
                        # ----------------------------------------------------------------------------------------------
                        # [3.0] vae decode
                        posterior = vae.encode(image).latent_dist
                        #mean, logvar = posterior.mean, posterior.logvar
                        for i in range(10):
                            # make many generator
                            generator = torch.Generator()
                            generator.manual_seed(i)
                            reconstruction = vae(sample = image, generator = generator,).sample # [1,3,512,512]
                            # autoencoder reconstruction image
                            image = image_processor.postprocess(reconstruction, output_type='pil')
                            image.save(f'{check_base_folder}/original_vae_{i}.png')

                        # ----------------------------------------------------------------------------------------------
                        # [3] stochastic sampling
                        z_mu, z_sigma = decoder_model.encode(hidden_latent)
                        trial_times = 10
                        for i in range(trial_times) :

                            z = decoder_model.sampling(z_mu, z_sigma)
                            reconstruction = decoder_model.decode(z)
                            # there is even minus value
                            reconstruction_img = reconstruction.squeeze(0).permute(1, 2, 0).detach().cpu()#.numpy()
                            # torch to numpy
                            np_img = np.array(((reconstruction_img + 1) / 2) * 255).astype(np.uint8)
                            pil = Image.fromarray(np_img)

                            #print(f'original value = {reconstruction_img}')
                            #plt.imshow((reconstruction_img * 255).astype(np.uint8))


                            # save
                            recon_folder = os.path.join(args.output_dir, 'reconstruct_folder')
                            os.makedirs(recon_folder, exist_ok=True)
                            #plt.savefig(f'{check_base_folder}/gen_lora_epoch_{lora_epoch}_{i}.png')
                            pil.save(f'{check_base_folder}/gen_lora_epoch_{lora_epoch}_{i}.png')
                            #plt.close()
        for k in raw_state_dict_orig.keys():
            raw_state_dict[k] = raw_state_dict_orig[k]
        network.load_state_dict(raw_state_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
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
    parser.add_argument("--do_semantic_position", action='store_true')
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
    parser.add_argument("--check_training", action='store_true')
    parser.add_argument("--pretrained_segmentation_model", type=str)
    parser.add_argument("--use_instance_norm", action='store_true')
    parser.add_argument("--aggregation_model_a", action='store_true')
    parser.add_argument("--aggregation_model_b", action='store_true')
    parser.add_argument("--aggregation_model_c", action='store_true')
    parser.add_argument("--mask_res", type=int, default=128)
    parser.add_argument("--pe_do_concat", action='store_true')

    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    main(args)