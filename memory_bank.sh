# !/bin/bash
# language 가 분명 작용하는듯 하다.

port_number=53333
category="medical"
obj_name="leader_polyp"
trigger_word="leader_polyp"
benchmark="Pranet"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="31_pvt_image_encoder_with_position_embedder_memorybank" #
# 3 --not_use_cls_token --without_condition except generation
# --gt_ext_npy \  --use_position_embedder
# 29_reducing_redundancy_without_noise_pred_with_position_embedding
# --reverse --reducing_redundancy
save_folder="31_pvt_image_encoder_with_position_embedder"
accelerate launch --config_file ../../gpu_config/gpu_0_config \
 --main_process_port $port_number memory_bank.py --log_with wandb \
 --output_dir "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --train_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/train" \
 --test_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/test" \
 --network_weights "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${save_folder}/model/lora-000143.safetensors" \
 --pretrained_segmentation_model "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${save_folder}/segmentation/segmentation-000143.pt" \
 --pretrained_vision_head "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${save_folder}/vision_head/vision-000143.pt" \
 --position_embedder_weights "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${save_folder}/position_embedder/position-000143.pt" \
 --network_dim 144 --network_alpha 4 \
 --resize_shape 512 \
 --latent_res 64 \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --n_classes 2 --mask_res 256 --batch_size 1 \
 --use_dice_ce_loss \
 --optimizer_args weight_decay=0.00005 \
 --use_image_condition \
 --image_processor 'pvt' \
 --image_model_training