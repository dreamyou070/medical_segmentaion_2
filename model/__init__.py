from model.lora import create_network
from model.pe import AllPositionalEmbedding, SinglePositionalEmbedding
from model.diffusion_model import load_target_model
import os
from safetensors.torch import load_file
from model.unet import TimestepEmbedding
from transformers import CLIPModel, ViTModel
def call_model_package(args, weight_dtype, accelerator, text_encoder_lora = True, unet_lora = True ):

    # [1] diffusion
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
    # [1.0] tes
    text_encoder.requires_grad_(False)
    # [1.1] vae
    vae.requires_grad_(False)
    vae.to(dtype=weight_dtype)
    vae.eval()
    # [1.2] unet
    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)

    # [2] image model
    if args.image_processor == 'clip':
        image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    elif args.image_processor == 'vit':
        image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    image_model = image_model.to(accelerator.device, dtype=weight_dtype)
    #if not args.image_model_training :
    #    image_model.eval()

    # [2] lora network
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value
    if args.use_image_condition :
        network = create_network(1.0, args.network_dim, args.network_alpha,
                                 vae, image_model, unet, neuron_dropout=args.network_dropout, **net_kwargs, )
    else :
        network = create_network(1.0, args.network_dim, args.network_alpha,
                                 vae, text_encoder, unet, neuron_dropout=args.network_dropout, **net_kwargs, )

    if args.use_text_condition :
        network.apply_to(text_encoder, unet, text_encoder_lora, unet_lora)
    else :
        network.apply_to(image_model, unet, apply_text_encoder=False, apply_unet=True)

    unet = unet.to(accelerator.device, dtype=weight_dtype)
    unet.eval()
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.eval()
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    if args.network_weights is not None :
        print(f' * loading network weights')
        info = network.load_weights(args.network_weights)
    network.to(weight_dtype)
    if args.use_text_condition :
        return text_encoder, vae, unet, network
    else :
        return image_model, vae, unet, network
