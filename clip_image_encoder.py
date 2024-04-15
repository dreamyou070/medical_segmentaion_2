from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
"""
clip_image_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


inputs = processor(images=image, return_tensors="pt", padding=True)#.data['pixel_values'] # [1,3,224,224]
pixel_values = inputs['data'].pixel_values
print(f'inputs 1 = {pixel_values.shape}')
clip_image_model.to("cuda:0")
image_inputs = inputs.to("cuda:0")

image_features = clip_image_model.get_image_features(**inputs) # [1, 768] (why not batchwize)

#print(f'image_features = {image_features.shape}')
"""

from transformers import AutoImageProcessor, ViTModel
import torch
# HOW TO ALIGNING TWO MODALITY ?

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
# [2] img
#image_path = 'data_sample/image/sample_200.jpg'
#image = Image.open(image_path)
#image_condition = image_processor(image, return_tensors="pt")
#image_condition['pixel_values'] = (image_condition['pixel_values']).squeeze()
#print(image_condition)
image_condition = {}
image_condition['pixel_values'] = torch.randn(1,4,64,64)
#p_value = image_condition['pixel_values'] # [1,3,224,224]
# add processor output
#pixel_values = inputs['data'].pixel_values
with torch.no_grad():
    outputs = model(**image_condition)

print(f'outputs = {outputs.last_hidden_state.shape}')
"""


last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
[1, 197, 768]
"""