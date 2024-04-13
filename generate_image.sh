# !/bin/bash

port_number=50052
category="medical"
obj_name="cardiac"
benchmark="acdc"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="Finetune_segment_head_feature_merging"
# --use_position_embedder \
# 64 / 4
# 128 / 64
accelerate launch --config_file ../../gpu_config/gpu_0_config \
 --main_process_port $port_number generate_image.py \
 --pretrained_model_name_or_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 \
 --data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/test" \
 --network_folder "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}/model" \
 --obj_name "${obj_name}" \
 --prompt "${trigger_word}" \
 --latent_res 64 \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --n_classes 4 \
 --mask_res 256 \
 --use_batchnorm --do_train_check