from model.blip import blip_decoder
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_size = 384
img_path = 'data_sample/image/sample_200.jpg'
raw_image = Image.open(img_path).convert('RGB')


transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
image = transform(raw_image).unsqueeze(0).to(device) # [batch sized ]


model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

with torch.no_grad():
    # beam search
    caption = model.generate(image,
                             sample=True,
                             num_beams=3, max_length=20, min_length=5)
    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
    print('caption: ' + caption[0])