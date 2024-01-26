map="5m_vs_6m"
algo_name="pref_qmix"
# # pos
# for i in {1..5}
# do
#     CUDA_VISIBLE_DEVICES=$cuda_device nohup python3 src/main.py \ 
#     --config=$alog_name --env-config=sc2 --map_name=$map \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=$i t_max=5050000 epsilon_anneal_time=1000000 \
#     save_model=False save_model_interval=200000 \
#     >/dev/null 2>&1 &
# done
# env_args.reward_sparse=False save_model=True save_model_interval=200000 
# for i in {2,3,4,5}
# do
#     CUDA_VISIBLE_DEVICES=$i nohup python3 src/main.py \
#     --config=qmix --env-config=sc2 --map_name=$map \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     env_args.reward_sparse=True \
#     >/dev/null 2>&1 &
# done
for i in {3,4}
do
    CUDA_VISIBLE_DEVICES=$i nohup python3 src/main.py \
    --config=$algo_name --env-config=sc2 --map_name=$map \
    with use_tensorboard=True env_args.reward_only_positive=True reward_lr=0.0003 \
    num_unsup_timesteps=1000 segment_size=10 max_feedback=100000 num_interact=3000 use_local_reward=True direct_local_preference=False \
    local_preference_type_training=indi_rewards local_preference_type=true_indi_rewards \
    lamda=0.01 lcoal_pretrain_timesteps=100000 reward_schedule=True t_max=2050000 use_global_reward=False \
    env_args.reward_sparse=False env_args.reward_indi_only_positive=False lamda_decay=False segment_capacity=5000 \
    >results/logs/5m_vs_6m_$i.log 2>&1 &
done
# TODO: policy exps
    # env_args.reward_sparse=True env_args.reward_indi_only_positive=True env_args.reward_scale=False \