import torch
encoder_hidden_states = torch.randn(2, 768)
encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
print(f'encoder_hidden_states = {encoder_hidden_states.shape}')