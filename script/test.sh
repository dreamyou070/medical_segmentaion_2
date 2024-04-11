# !/bin/bash

port_number=50042
category="medical"
obj_name="leader_polyp"
benchmark="bkai-igh-neopolyp"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="saving_test"

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../test.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/test" \
 --network_folder "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}/model" \
 --obj_name "${obj_name}" \
 --prompt "polyp" \
 --resize_shape 512 \
 --latent_res 64 \
 --use_position_embedder \
 --use_new_seg_unet \
 --norm_type "batch_norm" \
 --non_linearity "relu" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --image_folder_name 'image_256' --gt_folder_name 'mask_256' \
 --n_classes 3 \
 --mask_res 256