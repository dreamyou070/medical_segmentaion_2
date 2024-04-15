# text position is important
# 내가 성능을 볼 수 있다면 이유를 알고 찾아갈텐데...
# 그런 방법이 없으려나 ???
import torch
original_embedding = torch.randn((1,197,768))
cls_embedding = original_embedding[:,0,:]
image_embedding = original_embedding[:,1:,:]
# clustering
