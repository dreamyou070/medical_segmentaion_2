from PIL import Image
import numpy as np
import torch

def load_image(image_path, trg_h, trg_w):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    if trg_h and trg_w:
        image = image.resize((trg_w, trg_h), Image.BICUBIC)
    img = np.array(image, np.uint8)
    return img

@torch.no_grad()
def image2latent(image, vae, weight_dtype):
    if type(image) is Image:
        image = np.array(image)
    if type(image) is torch.Tensor and image.dim() == 4:
        latents = image
    else:
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(vae.device, weight_dtype)
        latents = vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
    return latents
