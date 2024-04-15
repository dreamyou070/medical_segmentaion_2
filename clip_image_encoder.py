from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

clip_image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

image_path = 'data_sample/image/sample_200.jpg'
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt", padding=True)#.data['pixel_values'] # [1,3,224,224]
pixel_values = inputs['data'].pixel_values
print(f'inputs 1 = {pixel_values.shape}')
clip_image_model.to("cuda:0")
image_inputs = inputs.to("cuda:0")

image_features = clip_image_model.get_image_features(**inputs) # [1, 768] (why not batchwize)

#print(f'image_features = {image_features.shape}')


from transformers import AutoImageProcessor, ViTModel
import torch

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

inputs = image_processor(image, return_tensors="pt")
pixel_values = inputs['data'].pixel_values
print(f'inputs 2 = {pixel_values.shape}')
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
[1, 197, 768]