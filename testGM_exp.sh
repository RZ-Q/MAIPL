# # 复现之前最好的
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=True actions_onehot=True reward_train_batch_size=128 loss_func=cross_entropy\
#     sample_episode_size=100 sample_segment_size=128 segment_size=10 segment_capacity=50000 \
#     local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/origin.log 2>&1 &
# # capacity
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=True actions_onehot=True reward_train_batch_size=128 loss_func=cross_entropy\
#     sample_episode_size=100 sample_segment_size=128 segment_size=10 segment_capacity=10000 \
#     local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/capacity.log 2>&1 &
# # segment size
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=True actions_onehot=True reward_train_batch_size=128 loss_func=cross_entropy\
#     sample_episode_size=100 sample_segment_size=128 segment_size=15 segment_capacity=50000 \
#     local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/segment.log 2>&1 &
# # policy
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=True actions_onehot=True reward_train_batch_size=128 loss_func=cross_entropy\
#     sample_episode_size=100 sample_segment_size=128 segment_size=10 segment_capacity=50000 global_preference_type=policy \
#     local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/policy.log 2>&1 &
# # es ss
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=True actions_onehot=True reward_train_batch_size=128 loss_func=cross_entropy\
#     sample_episode_size=150 sample_segment_size=200 segment_size=10 segment_capacity=50000 \
#     local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/es_ss.log 2>&1 &
# # reward schedule
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=True actions_onehot=True reward_train_batch_size=128 loss_func=cross_entropy\
#     sample_episode_size=100 sample_segment_size=128 segment_size=10 segment_capacity=50000 reward_schedule=True \
#     local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/schedule.log 2>&1 &
# # local preference
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=True actions_onehot=True reward_train_batch_size=128 loss_func=cross_entropy\
#     sample_episode_size=100 sample_segment_size=128 segment_size=10 segment_capacity=50000 \
#     local_script_preference=True lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/local.log 2>&1 &
# # KL
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=True actions_onehot=True reward_train_batch_size=128 loss_func=KL\
#     sample_episode_size=100 sample_segment_size=128 segment_size=10 segment_capacity=50000 \
#     local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/KL.log 2>&1 &
# # obs
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     num_seed_timesteps=2000 num_unsup_timesteps=10000 num_interact=150 reset_critic=True \
#     active=tan reward_lr=0.0003 reward_hidden_size=512 ensemble_size=3 \
#     state_or_obs=False actions_onehot=True reward_train_batch_size=128 loss_func=cross_entropy\
#     sample_episode_size=100 sample_segment_size=128 segment_size=10 segment_capacity=50000 \
#     local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
#     >results/logs/obs.log 2>&1 &
# 再跑一次 RM
CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
    --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    num_seed_timesteps=2000 num_unsup_timesteps=10000 \
    active=tan reward_lr=0.0003 reward_hidden_size=256 ensemble_size=3 max_feedback=20000 \
    state_or_obs=False seed=180336608\
    segment_size=10 segment_capacity=50000 \
    local_script_preference=False lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
    >results/logs/origin.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
    --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    num_seed_timesteps=2000 num_unsup_timesteps=10000 \
    active=tan reward_lr=0.0003 reward_hidden_size=256 ensemble_size=3 max_feedback=20000 \
    state_or_obs=False seed=180336608\
    segment_size=10 segment_capacity=50000 \
    local_script_preference=True lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.005 \
    >results/logs/0.005.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
    --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    num_seed_timesteps=2000 num_unsup_timesteps=10000 \
    active=tan reward_lr=0.0003 reward_hidden_size=256 ensemble_size=3 max_feedback=20000 \
    state_or_obs=False seed=180336608\
    segment_size=10 segment_capacity=50000 \
    local_script_preference=True lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.001 \
    >results/logs/0.001.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
    --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    num_seed_timesteps=2000 num_unsup_timesteps=10000 \
    active=tan reward_lr=0.0003 reward_hidden_size=256 ensemble_size=3 max_feedback=20000 \
    state_or_obs=False seed=180336608\
    segment_size=10 segment_capacity=50000 \
    local_script_preference=True lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.01 \
    >results/logs/0.01.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
    --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    num_seed_timesteps=2000 num_unsup_timesteps=10000 \
    active=tan reward_lr=0.0003 reward_hidden_size=256 ensemble_size=3 max_feedback=20000 \
    state_or_obs=False\
    segment_size=10 segment_capacity=50000 \
    local_script_preference=True lcoal_pretrain_timesteps=200000 local_preference_type=true_indi_rewards lamda=0.0005 \
    >results/logs/0.0005.log 2>&1 &