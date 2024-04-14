import torch, os

trg_attention = torch.randn(64*64,2)
max_value = torch.max(trg_attention, dim=0).values
print(max_value)

text_predict = torch.where(max_value>0.5, 0, 1) # erase index
key_word_index = torch.tensor([10,14])          # erase index
key_word_index = key_word_index * text_predict
key_word_index = torch.tensor([10,0])
print(key_word_index)
