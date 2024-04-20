

import torch

original_tensor = torch.randn((1, 197, 768))
cls_tensor = original_tensor[:, 0, :] # [1,1,768]
text_tensor = original_tensor[:, 1:, :]

class_1_tensor = torch.randn((1,768))
class_2_tensor = torch.randn((1,768))

# feature reduction
new_features = set()

new_features.add(cls_tensor.unsqueeze(0))

for pix_i in range(text_tensor.shape[1]):
    feature = text_tensor[:, pix_i, :]
    class_1_sim = torch.nn.functional.cosine_similarity(feature, class_1_tensor)
    class_2_sim = torch.nn.functional.cosine_similarity(feature, class_2_tensor)
    if class_1_sim.item() > class_2_sim.item():
        mean_feat = class_1_tensor.unsqueeze(0)
    else:
        mean_feat = class_2_tensor.unsqueeze(0)
    new_features.add(mean_feat)
print(len(new_features))
"""
new_features = list(new_features)
new_features = torch.stack(new_features, dim=1)
#print(new_features.shape) # [1, 197, 768]
"""