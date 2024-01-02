# CUDA_VISIBLE_DEVICxES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 \
#     >results/logs/qdiffer_wGR_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=False \
#     seed=2 \
#     >results/logs/qdiffer_wGR_5mvs6m_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=False \
#     seed=3 \
#     >results/logs/qdiffer_wGR_5mvs6m_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=False \
#     seed=4 \
#     >results/logs/qdiffer_wGR_5mvs6m_4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qmix_w_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 \
#     >results/logs/qmix_w_globalRM_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=qmix_w_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=2 \
#     >results/logs/qmix_w_globalRM_5mvs6m_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=qmix_w_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=3 \
#     >results/logs/qmix_w_globalRM_5mvs6m_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=qmix_w_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=4 \
#     >results/logs/qmix_w_globalRM_5mvs6m_4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 local_script_preference=True lamda=0.001 \
#     >results/logs/qdiffer_wGR_local001_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 local_script_preference=True lamda=0.005 \
#     >results/logs/qdiffer_wGR_local005_5mvs6m_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 local_script_preference=True lamda=0.01 \
#     >results/logs/qdiffer_wGR_local01_5mvs6m_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=qdiffer_globalRM --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 local_script_preference=True lamda=0.0005 \
#     >results/logs/qdiffer_wGR_local0005_5mvs6m_4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 use_local_reward=False \
#     >results/logs/pref_qmix_nolocal_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=2 use_local_reward=False \
#     >results/logs/pref_qmix_nolocal_5mvs6m_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=3 use_local_reward=False \
#     >results/logs/pref_qmix_nolocal_5mvs6m_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=4 use_local_reward=False \
#     >results/logs/pref_qmix_nolocal_5mvs6m_4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=2 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_5mvs6m_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=3 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_5mvs6m_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=4 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_5mvs6m_4.log 2>&1 &
# add thres
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_thres05_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=2 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_thres05__5mvs6m_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=3 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_thres05__5mvs6m_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=4 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_thres05__5mvs6m_4.log 2>&1 &
# use between seg reward model
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 use_local_reward=True lamda=0.001 \
#     >results/logs/pref_qmix_local001_betw_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 use_local_reward=True lamda=0.003 \
#     >results/logs/pref_qmix_local003_betw_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 use_local_reward=True lamda=0.005 \
#     >results/logs/pref_qmix_local005_betw_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 use_local_reward=True lamda=0.0005 \
#     >results/logs/pref_qmix_local0005_betw_5mvs6m_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=pref_qmix --env-config=sc2 --map_name=5m_vs_6m \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=1 use_local_reward=True lamda=0.01 \
#     >results/logs/pref_qmix_local01_betw_5mvs6m_1.log 2>&1 &
# aloha
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qmix --env-config=aloha --map_name=aloha \
#     with use_tensorboard=True env_args.punishment=0 \
#     seed=1 save_model=True save_model_interval=500000 \
#     >results/logs/qmix_aloha_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=qmix --env-config=aloha --map_name=aloha \
#     with use_tensorboard=True env_args.punishment=0 \
#     seed=2 save_model=True save_model_interval=500000 \
#     >results/logs/qmix_aloha_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=qmix --env-config=aloha --map_name=aloha \
#     with use_tensorboard=True env_args.punishment=0 \
#     seed=3 save_model=True save_model_interval=500000 \
#     >results/logs/qmix_aloha_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=qmix --env-config=aloha --map_name=aloha \
#     with use_tensorboard=True env_args.punishment=0 \
#     seed=4 save_model=True save_model_interval=500000 \
#     >results/logs/qmix_aloha_4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py \
#     --config=qmix --env-config=aloha --map_name=aloha \
#     with use_tensorboard=True env_args.punishment=10 \
#     seed=1 \
#     >results/logs/qmix_aloha_punish_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 src/main.py \
#     --config=qmix --env-config=aloha --map_name=aloha \
#     with use_tensorboard=True env_args.punishment=10 \
#     seed=2 \
#     >results/logs/qmix_aloha_punish_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 src/main.py \
#     --config=qmix --env-config=aloha --map_name=aloha \
#     with use_tensorboard=True env_args.punishment=10 \
#     seed=3 \
#     >results/logs/qmix_aloha_punish_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python3 src/main.py \
#     --config=qmix --env-config=aloha --map_name=aloha \
#     with use_tensorboard=True env_args.punishment=10 \
#     seed=4 \
#     >results/logs/qmix_aloha_punish_4.log 2>&1 &