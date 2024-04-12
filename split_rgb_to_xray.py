import os
from PIL import Image
import numpy as np

img_path = r'gen_lora_epoch_000007_6.png'
rgb_np = np.array(Image.open(img_path).convert('RGB'))
r,g,b = rgb_np[:,:,0], rgb_np[:,:,1], rgb_np[:,:,2]
r = Image.fromarray(r).convert('L')
g = Image.fromarray(g).convert('L')
b = Image.fromarray(b).convert('L')

r.save('r.png')
g.save('g.png')
b.save('b.png')