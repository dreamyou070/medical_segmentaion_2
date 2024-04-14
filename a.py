import torch, os

trg_attention = torch.randn(64*64,2)
max_value = torch.max(trg_attention, dim=0).values
print(max_value)

origin_sentence_ids = torch.Tensor([[[49406,   589,  1674,   533,   539,   321,   267,   335,   267,   333,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407]]])
erase_idx = int(torch.Tensor([6]).item())
# erase only the 6th index
re_pre_index = origin_sentence_ids[:,:,:erase_idx]
re_post_index = origin_sentence_ids[:,:,erase_idx+1:]
re_index = torch.cat((re_pre_index, re_post_index), dim=-1)

import numpy as np

a = np.array([1,2])
print(a.shape[0])