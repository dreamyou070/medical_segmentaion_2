#! /bin/bash

bench_mark="MVTec"
class_name='zipper'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="sigma_max_60_min_sigma_25_max_perlin_scale_6_max_beta_scale_0.6_min_beta_scale_0_not_rot"

base_save_dir="../../result/${bench_mark}/${class_name}/${layer_name}/${sub_folder}/${file_name}"

python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir ${base_save_dir}