from transformers import CLIPTokenizer
import os

TOKENIZER_PATH = "openai/clip-vit-large-patch14"

def load_tokenizer(args):
    original_path = TOKENIZER_PATH
    tokenizer: CLIPTokenizer = None
    tokenizer = CLIPTokenizer.from_pretrained(original_path)
    return tokenizer