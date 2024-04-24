import torch
feat = torch.randn(160)
memory_bank = [feat, feat, feat]
memory_bank = torch.stack(memory_bank)
print(memory_bank.size())

sample = torch.randn(1,160,256,256)
x = mean + std * sample
print(x.size())