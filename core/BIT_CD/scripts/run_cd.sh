#!/usr/bin/env bash

gpus=0
checkpoint_root=ckpt_dir
data_name=PVP-India

img_size=256
batch_size=8
lr=1e-3
max_epochs=100
net_G=base_transformer_pos_s4_dd8
#base_resnet18
#base_transformer_pos_s4_dd8
#base_transformer_pos_s4_dd8_dedim8
lr_policy=linear

split=train
split_val=val
# project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${ssplit_val}_${max_epochs}_${lr_policy}
project_name=0702_PVP_tent
loss=unsuper

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}  --loss ${loss}