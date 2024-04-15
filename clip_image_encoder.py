from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

clip_image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

image_path = 'data_sample/image/sample_200.jpg'
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt", padding=True)#.data['pixel_values'] # [1,3,224,224]
clip_image_model.to("cuda:0")
image_inputs = inputs.to("cuda:0")

image_features = clip_image_model.get_image_features(**inputs) # [1, 768] (why not batchwize)

print(f'image_features = {image_features.shape}')