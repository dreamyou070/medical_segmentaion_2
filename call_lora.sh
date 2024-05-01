# !/bin/bash
# 기존 논문들에서는 loss 를 2가지 이상을 사용하고 있구나 ?
#
port_number=58852
category="medical"
obj_name="leader_polyp"
trigger_word="leader_polyp"
benchmark="Pranet_Sub"
layer_name='layer_3'
sub_folder="up_16_32_64_20240501"
file_name="3_class_0_pvt_image_encoder" #

accelerate launch --config_file ../../gpu_config/gpu_0_config \
 --main_process_port $port_number call_lora.py --log_with wandb \
 --output_dir "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --train_data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/train/class_0" \
 --base_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}" \
 --network_dim 64 --network_alpha 4 --resize_shape 512 --latent_res 64 --trigger_word "${trigger_word}" --obj_name "${obj_name}"