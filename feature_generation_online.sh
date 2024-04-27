# !/bin/bash
port_number=53922
category="camouflaged"
obj_name="CAMO"
trigger_word="CAMO"
benchmark="CAMO-V.1.0-CVIU2019_sy"
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="2_basis_pvt" # best 0.852

accelerate launch --config_file ../../gpu_config/gpu_0_config \
 --main_process_port $port_number feature_generation_online.py --log_with wandb \
 --output_dir "../result/${category}/${obj_name}/${benchmark}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 200 \
 --pretrained_model_name_or_path ../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --train_data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/train" \
 --test_data_path "/home/dreamyou070/MyData/anomaly_detection/${category}/${obj_name}/${benchmark}/test" \
 --network_dim 144 --network_alpha 4 --resize_shape 512 --latent_res 64 --trigger_word "${trigger_word}" --obj_name "${obj_name}" \
 --trg_layer_list "['up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --n_classes 2 --mask_res 256 --batch_size 1 \
 --use_dice_ce_loss \
 --optimizer_args weight_decay=0.00005 \
 --use_image_condition \
 --image_model_training --image_processor 'pvt' \
 --use_position_embedder \
 --anomal_mse_loss --online_pseudo_loss \
 --use_positioning_module --use_channel_attn \
 --use_simple_segmodel