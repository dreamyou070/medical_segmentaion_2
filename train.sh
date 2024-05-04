# !/bin/bash
# 기존 논문들에서는 loss 를 2가지 이상을 사용하고 있구나 ?
#
port_number=58800
category="medical"
obj_name="leader_polyp"
trigger_word="leader_polyp"
benchmark="Pranet_Sub0"
layer_name='layer_3'
sub_folder="up_16_32_64_20240501"
file_name="5_base_class_0" #

accelerate launch --config_file ../../gpu_config/gpu_0_config \
 --main_process_port $port_number train.py --log_with wandb \
 --output_dir "../result/${category}/${obj_name}/Pranet_Sub/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 8 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --train_data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/train/res_256" \
 --base_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/Pranet" \
 --network_dim 64 --network_alpha 4 --resize_shape 512 --latent_res 64 --trigger_word "${trigger_word}" --obj_name "${obj_name}" \
 --network_weights "../result/${category}/${obj_name}/Pranet_Sub/${sub_folder}/${file_name}/model/lora-000008.safetensors" \
 --segmentation_model_weights "../result/${category}/${obj_name}/Pranet_Sub/${sub_folder}/${file_name}/segmentation/segmentation-000008.pt" \
 --vision_head_weights "../result/${category}/${obj_name}/Pranet_Sub/${sub_folder}/${file_name}/vision_head/vision-000008.pt" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --n_classes 2 --mask_res 64 --batch_size 1 \
 --use_dice_ce_loss --optimizer_args weight_decay=0.00005 \
 --use_image_condition --image_model_training --image_processor 'pvt' --reverse \
 --use_simple_segmodel --use_segmentation_model