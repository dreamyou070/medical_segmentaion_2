# !/bin/bash
# language 가 분명 작용하는듯 하다.

port_number=54166
category="medical"
obj_name="leader_polyp"
trigger_word="leader_polyp"
benchmark="Pranet"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="3_Finetune_segment_not_gen_per_image_caption"

accelerate launch --config_file ../../gpu_config/gpu_0_1_2_3_config \
 --main_process_port $port_number feature_generator.py --log_with wandb \
 --output_dir "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --train_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/train" \
 --test_data_path "/home/dreamyou070/MyData/anomaly_detection/medical/${obj_name}/${benchmark}/test" \
 --network_dim 64 --network_alpha 4 \
 --resize_shape 512 \
 --latent_res 64 \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --n_classes 2 \
 --mask_res 256 \
 --use_batchnorm \
 --use_dice_ce_loss \
 --optimizer_args weight_decay=0.00005 \
 --init_latent_p 1.0 \
 --generator_loss_weight 1.0 \
 --segmentation_loss_weight 1.0 \
 --use_image_by_caption