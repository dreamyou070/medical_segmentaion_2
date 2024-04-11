import os, torch
from safetensors.torch import save_file
def save_model(args, saving_folder, saving_name, unwrapped_nw, save_dtype):

    # [1] state dictionary
    metadata = {}
    state_dict = unwrapped_nw.state_dict()
    for key in list(state_dict.keys()):
        v = state_dict[key]
        v = v.detach().clone().to("cpu").to(save_dtype)
        state_dict[key] = v

    # [2] saving
    save_model_base_dir = os.path.join(args.output_dir, saving_folder)
    os.makedirs(save_model_base_dir, exist_ok=True)
    if os.path.splitext(saving_name)[1] == ".safetensors":
        save_file(state_dict, os.path.join(save_model_base_dir, saving_name), metadata)
    else :
        torch.save(state_dict, os.path.join(save_model_base_dir, saving_name))