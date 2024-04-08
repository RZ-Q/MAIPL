#!/bin/sh
map="2c_vs_64zg"
model_type="MR"
quality=("good" "medium" "poor" "good_medium" "medium_poor")
reward_suffix=('55' '85' '95' '40' '65')
for i in {0,1,2,3,4}
do
    qua_=${quality[$i]}
    rs_=${reward_suffix[$i]}
    CUDA_VISIBLE_DEVICES=6 python3 reference_reward_model.py \
        --env=smac \
        --task=$map \
        --qua=$qua_\
        --reward_suffix=$rs_ \
        --model_type=$model_type \
        >1.log 2>&1 &
done
