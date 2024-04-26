import torch
memory_iter = 31

parameters = torch.randn(10,160)
mean = parameters.mean(dim=0).unsqueeze(0)
var = parameters.var(dim=0).unsqueeze(0)
print(mean.shape)