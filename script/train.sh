# !/bin/bash
#
port_number=50010
category="medical"
obj_name="leader_polyp"
benchmark="bkai-igh-neopolyp"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="3_segmentation_model_a_cross_focal_layer_norm_relu_replay_experiment"
# [1] data info
train_data_path="/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/train"
test_data_path="/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/test"
resize_shape=512
latent_res=64
mask_res=256
n_classes=3
trigger_word="polyp"

# [2] segmentation  info

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_5_config \
 --main_process_port $port_number ../train.py --log_with wandb \
 --output_dir "../../result/${category}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --train_data_path "${train_data_path}" \
 --test_data_path "${test_data_path}" \
 --resize_shape ${resize_shape} \
 --latent_res ${latent_res} \
 --mask_res ${mask_res} \
 --n_classes ${n_classes} \
 --trigger_word "${trigger_word}" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --use_position_embedder \
 --aggregation_model_a --norm_type "layer_norm" --non_linearity "relu"