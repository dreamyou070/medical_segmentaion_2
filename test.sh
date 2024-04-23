# !/bin/bash

port_number=50052
category="medical"
obj_name="leader_polyp"
benchmark="Pranet"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="31_pvt_image_encoder_with_position_embedder"
# --use_position_embedder \
# 64 / 4
# 128 / 64
accelerate launch --config_file ../../gpu_config/gpu_0_config \
 --main_process_port $port_number test.py \
 --pretrained_model_name_or_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 144 --network_alpha 4 \
 --network_weights "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}/model/lora-000112.safetensors" \
 --segmentation_head_weights "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}/segmentation/segmentation-000112.pt" \
 --output_dir "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}" \
 --base_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/test" \
 --obj_name "${obj_name}" \
 --latent_res 64 \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --n_classes 2 \
 --use_position_embedder \
 --mask_res 256 --use_image_condition --reducing_redundancy --image_processor 'pvt'