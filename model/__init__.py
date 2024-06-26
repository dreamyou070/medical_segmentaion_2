from model.lora import create_network
from model.pe import AllPositionalEmbedding, SinglePositionalEmbedding
from model.diffusion_model import load_target_model
from model.unet import TimestepEmbedding
from model.modeling_vit import ViTModel
import torch
from polyppvt.lib.pvt import PolypPVT
def call_model_package(args, weight_dtype, accelerator, text_encoder_lora = True, unet_lora = True ):

    # [1] diffusion
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
    del text_encoder

    # [1.1] vae
    vae.requires_grad_(False)
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()

    # [1.2] unet
    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)

    # [1.3] lora network
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value

    if args.image_processor == 'vit': # ViTModel
        image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    elif args.image_processor == 'pvt':
        model = PolypPVT()
        pretrained_pth_path = '/share0/dreamyou070/dreamyou070/PolypPVT/Polyp_PVT/model_pth/PolypPVT.pth'
        model.load_state_dict(torch.load(pretrained_pth_path))
        image_model = model.backbone  # pvtv2_b2 model

    image_model = image_model.to(accelerator.device,dtype=weight_dtype)
    image_model.requires_grad_(False)

    # [1.4]
    condition_model = image_model # image model is a condition
    condition_modality = 'image'

    """ see well how the model is trained """
    network = create_network(1.0,
                             args.network_dim,
                             args.network_alpha,
                             vae,
                             condition_model=condition_model,
                             unet=unet,
                             neuron_dropout=args.network_dropout,
                             condition_modality=condition_modality,
                             **net_kwargs, )
    network.apply_to(condition_model,
                     unet,
                     True,
                     True,
                     condition_modality=condition_modality)

    if args.network_weights is not None :
        print(f' * loading network weights')
        info = network.load_weights(args.network_weights)
    network.to(dtype = weight_dtype, device = accelerator.device)

    # [1.2] unet
    #unet = unet.to(accelerator.device, dtype=weight_dtype)
    #unet.eval()
    # [1.3] network
    #condition_model = condition_model.to(accelerator.device, dtype=weight_dtype)
    #condition_model.eval()


    return condition_model, vae, unet, network, condition_modality
