from transformers import pipeline
from PIL import Image
import requests

# [1] get image
image_path = 'data_sample/image/sample_200.jpg'
image = Image.open(image_path).convert('RGB')

# [2] make pipeline
semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
results = semantic_segmentation(image)