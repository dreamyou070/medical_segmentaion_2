import os
import numpy as np
from torch.utils.data import Dataset
import torch
from transformers import CLIPTokenizer

class_map = {0: ['b', 'brain'],
             1: ['n', 'non-enhancing tumor core'],
             2: ['e', 'edema'],
             3: ['t', 'enhancing tumor'], }

base_prompts = ['this is a picture of ',
                'this is a picture of a ',
                'this is a picture of an ',
                'this is a picture of the ',
                'this picture is of ', ]
"""
TOKENIZER_PATH = "openai/clip-vit-large-patch14"

def load_tokenizer(args):
    original_path = TOKENIZER_PATH
    tokenizer: CLIPTokenizer = None
    tokenizer = CLIPTokenizer.from_pretrained(original_path)
    return tokenizer



class_es = [1,3]
n_classes = 4

# [1] caption
caption = base_prompts[np.random.randint(0, len(base_prompts))]
for i, class_idx in enumerate(class_es):
    caption += class_map[class_idx][0]
    if i < len(class_es) - 1:
        caption += ', '
# final caption = 'this is a picture of b, n, e, t'

# [2] key words
key_words = [class_map[i][0] for i in class_es]

# [3] target indexs
tokenizer = load_tokenizer(None)
caption_token = tokenizer(caption, padding="max_length", truncation=True, return_tensors="pt")
input_ids = caption_token.input_ids
attention_mask = caption_token.attention_mask

def get_target_index(target_words, caption):

    target_word_index = []
    for target_word in target_words:
        target_word_token = tokenizer(target_word, return_tensors="pt")
        target_word_input_ids = target_word_token.input_ids[:, 1]
        # [1] get target word index from sentence token
        sentence_token = tokenizer(caption, return_tensors="pt")
        sentence_token = sentence_token.input_ids
        batch_size = sentence_token.size(0)
        for i in range(batch_size):
            # same number from sentence token to target_word_inpud_ids
            s_tokens = sentence_token[i]
            idx = (torch.where(s_tokens == target_word_input_ids))[0].item()
            target_word_index.append(idx)
    return target_word_index
# list extend
"""
attn_maps = torch.randn(1, 2, 256,256)
n_classes = 4


print(masks_pred.shape) # [1,4,256,256]