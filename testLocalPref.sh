# # qmix
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     >results/logs/qmix.log 2>&1 &
# # qdiffer
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qmix_differ --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     >results/logs/qmix_differ.log 2>&1 &
# pref qmix w/o rm only local pref
CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
    --config=pref_qdiffer_wo_RM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    lamda=0.5 \
    >results/logs/local_lamda0.5.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
    --config=pref_qdiffer_wo_RM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    lamda=0.1 \
    >results/logs/local_lamda0.1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
    --config=pref_qdiffer_wo_RM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    lamda=0.05 \
    >results/logs/local_lamda0.05.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
    --config=pref_qdiffer_wo_RM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    lamda=0.01 \
    >results/logs/local_lamda0.01.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
    --config=pref_qdiffer_wo_RM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    lamda=0.005 \
    >results/logs/local_lamda0.005.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
    --config=pref_qdiffer_wo_RM --env-config=sc2 --map_name=5m_vs_6m \
    with use_tensorboard=True env_args.reward_only_positive=True \
    lamda=0.001 \
    >results/logs/local_lamda0.001.log 2>&1 &