from transformers import pipeline
from PIL import Image
import requests
import torch
from transformers import AutoImageProcessor, SegformerModel
import torch
from transformers import SegformerImageProcessor, SegformerForImageClassification
# from transformers import SegformerImageProcessor, SegformerForImageClassification
from torch import nn


"""    

c = 320

def self.dim_and_res_up(mlp_layer, upsample_layer, x) :
    # [batch, dim, res, res] -> [batch, res*res, dim]
    batch, dim, res, res = x.shape
    x = x.permute(0, 2, 3, 1).contious().reshape(1, res*res, dim)
    # [1] dim change
    x = mlp_layer(x) # x = [batch, res*res, new_dim]
    new_dim = x.shape[-1]
    x = x.permute(0, 2, 1).contious().reshape(1, new_dim, res, res)
    # [2] res change
    x = upsample_layer(x)
    return x

mlp_layer_1 = torch.nn.Linear(1280, c)
mlp_layer_2 = torch.nn.Linear(640, c)
mlp_layer_3 = torch.nn.Linear(320, c)
upsample_layer_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
upsample_layer_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
upsample_layer_3 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)

x16_out = dim_and_res_up(mlp_layer_1, upsample_layer_1, x16_out)
x32_out = dim_and_res_up(mlp_layer_2, upsample_layer_2, x32_out)
x64_out = dim_and_res_up(mlp_layer_3, upsample_layer_3, x64_out)
"""
"""
# [1] get image
image_path = 'data_sample/image/sample_200.jpg'
image = Image.open(image_path).convert('RGB')

# [2] make pipeline
image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
inputs = image_processor(images=image, return_tensors="pt") # shape of pixel_values: torch.Size([1, 3, 512,512])


# [3] main model
# mit-b0 is base model and not for segmentation model
# model = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
# SegformerForSemanticSegmentation
semantic_segmentation_0 = pipeline(task = "image-segmentation", model = "nvidia/mit-b0").model
semantic_segmentation_1 = pipeline(task = "image-segmentation", model = "nvidia/mit-b1").model
semantic_segmentation_2 = pipeline(task = "image-segmentation", model = "nvidia/mit-b2").model
semantic_segmentation_3 = pipeline(task = "image-segmentation", model = "nvidia/mit-b3").model
semantic_segmentation_4 = pipeline(task = "image-segmentation", model = "nvidia/mit-b4").model
semantic_segmentation_5 = pipeline(task = "image-segmentation", model = "nvidia/mit-b5").model

out_0 = semantic_segmentation_0(pixel_values = inputs['pixel_values'])
out_1 = semantic_segmentation_1(pixel_values = inputs['pixel_values'])
out_2 = semantic_segmentation_2(pixel_values = inputs['pixel_values'])
out_3 = semantic_segmentation_3(pixel_values = inputs['pixel_values'])
out_4 = semantic_segmentation_4(pixel_values = inputs['pixel_values'])
out_5 = semantic_segmentation_5(pixel_values = inputs['pixel_values'])
"""
"""

output = model(pixel_values = inputs['pixel_values'])

print(output.logits.shape)

# [3] segformer model
#input_torch = torch.randn(1, 3, 1024, 1024)
# SegformerForSemanticSegmentation

#
#from datasets import load_dataset

#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]


model = SegformerModel.from_pretrained("nvidia/mit-b0")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
[1, 256, 16, 16]
"""