#!/bin/sh
env="StarCraft2"
map="corridor"
model_type="MultiPrefTransformerDivide"
seed=1
max_traj_length=32
data_dir="/data/user/kouqian/files/MAOfflineDatasets/SMAC410_pref/"
suffix=".pkl"

echo "env is ${env}, task is ${map}, model_type is ${model_type}, exp is ${map}, seed is ${seed}"
# for dp in {"_good_5000_32","_good_medium_5000_32","_medium_5000_32","_medium_poor_5000_32","_poor_5000_32"}
# for dp in {"_good_5000_32","_good_medium_5000_32"}
# for dp in {"_medium_5000_32","_medium_poor_5000_32","_poor_5000_32"}
# for dp in {"_medium_5000_32","_medium_poor_5000_32"}
# for dp in {"_good_medium_5000_32","_poor_5000_32"}
dp="_poor_5000_32"
# do
data_path="$data_dir$map$dp$suffix"
# MultiPrefTransformerDivide
CUDA_VISIBLE_DEVICES=6 python3 train_reward_model.py \
    --comment ${dp} \
    --multi_transformer.embd_dim 256 --multi_transformer.action_embd_dim 64 \
    --multi_transformer.n_layer 1 --multi_transformer.n_head 4 \
    --multi_transformer.use_dropout True --multi_transformer.use_lstm True \
    --multi_transformer.use_weighted_sum=True \
    --batch_size 256 --n_epochs 100 --seed ${seed} --model_type ${model_type} \
    --max_traj_length ${max_traj_length} --env "smac" --task ${map} \
    --dataset_path=$data_path \
    >/dev/null 2>&1 &

# PrefTransformer
# CUDA_VISIBLE_DEVICES=5 python3 train_reward_model.py \
#     --comment ${dp} \
#     --transformer.use_weighted_sum True --transformer.train_type sum\
#     --batch_size 256 --n_epochs 100 --seed ${seed} --model_type ${model_type} \
#     --max_traj_length ${max_traj_length} --env "smac" --task ${map} \
#     --dataset_path=$data_path \
#     >/dev/null 2>&1 &

# NMR & MR
# CUDA_VISIBLE_DEVICES=3 python3 train_reward_model.py \
#     --comment ${dp} \
#     --batch_size 256 --n_epochs 100 --seed ${seed} --model_type ${model_type} \
#     --max_traj_length ${max_traj_length} --env "smac" --task ${map} \
#     --dataset_path=$data_path \
#     >1.log 2>&1 &
# done
