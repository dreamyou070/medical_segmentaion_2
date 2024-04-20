import torch
from PIL import Image
from torchvision import transforms

from model.resnet import resnet101, ResNet101_Weights
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
#model.eval()
model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
"""

# [2] image
image_path = r'data_sample/image/sample_200.jpg'
input_image = Image.open(image_path)

preprocess = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# [3] get feature
# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0].shape)
"""