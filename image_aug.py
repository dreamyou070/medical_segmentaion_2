import os
from PIL import Image
import numpy as np

image_dir = 'image_sample_200.jpg'
pil = Image.open(image_dir).convert('RGB').resize((512, 512))
# rotate
pil = pil.rotate(90)
#pil.show()

# mask
mask_dir = 'sample_200.npy'
mask = np.load(mask_dir)
# rotate
mask = np.rot90(mask)
mask = np.rot180(mask)
mask = np.rot270(mask)