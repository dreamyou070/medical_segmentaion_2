from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_path = 'data_sample/image/sample_200.jpg'
image = Image.open(image_path)
image_inputs = processor(images=image, return_tensors="pt", padding=True).data
print(f'image_inputs = {image_inputs}')