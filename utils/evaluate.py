from utils import reshape_batch_dim_to_heads
import torch
import numpy as np
from utils.evaluate import generate_confusion_matrix, get_IoU

@torch.inference_mode()
def evaluation_check(segmentation_head, dataloader, device, text_encoder, unet, vae, controller, weight_dtype,
                     position_embedder, args):
    segmentation_head.eval()

    with torch.no_grad():
        y_true_list, y_pred_list = [], []
        for global_num, batch in enumerate(dataloader):
            with torch.set_grad_enabled(True):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
            image = batch['image'].to(dtype=weight_dtype)                                   # 1,3,512,512
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)                               # 1,128*128
            gt = batch['gt'].to(dtype=weight_dtype)                                         # 1,4,128,128
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
            with torch.set_grad_enabled(True):
                unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=position_embedder)
            query_dict, key_dict = controller.query_dict, controller.key_dict
            controller.reset()
            q_dict = {}
            for layer in args.trg_layer_list:
                query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                res = int(query.shape[1] ** 0.5)
                q_dict[res] = reshape_batch_dim_to_heads(query) # 1, res,res,dim
            x16_out, x32_out, x64_out = q_dict[16], q_dict[32], q_dict[64]
            masks_pred = segmentation_head(x16_out, x32_out, x64_out) # 1,4,128,128
            # [1] pred
            mask_pred_ = masks_pred.permute(0, 2, 3, 1).detach().cpu().numpy()  # 1,128,128,4
            mask_pred_argmax = np.argmax(mask_pred_, axis=3).flatten()  # 128*128
            mask_pred_argmax = torch.nn.functional.one_hot(torch.Tensor(mask_pred_argmax).to(torch.int64),
                                                           num_classes=args.n_classes)
            y_pred_list.append(mask_pred_argmax)
            y_true_list.append(gt_flat.flatten().squeeze())
    y_hat = torch.cat(y_pred_list).to('cpu')
    y = torch.cat(y_true_list).to('cpu').long()
    confusion_matrix, dice_coeff = generate_confusion_matrix(y_pred = y_hat,
                                                             y_true = y,
                                                             class_num = args.n_classes)
    IoU_dict, mIOU = get_IoU(y_hat, y, args.n_classes, confusion_matrix)
    segmentation_head.train()
    return IoU_dict, confusion_matrix, dice_coeff