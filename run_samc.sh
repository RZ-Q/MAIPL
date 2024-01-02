map="6h_vs_8z"
cuda_device=3
alog_name="qplex"
# pos
for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$cuda_device nohup python3 src/main.py \
    --config=qmix --env-config=sc2 --map_name=$map \
    with use_tensorboard=True env_args.reward_only_positive=True \
    seed=$i t_max=5050000 epsilon_anneal_time=1000000 \
    save_model=False save_model_interval=200000 \
    >/dev/null 2>&1 &
done
# do
#     CUDA_VISIBLE_DEVICES=$cuda_device nohup python3 src/main.py \
#     --config=qmix --env-config=sc2 --map_name=$map \
#     with use_tensorboard=True env_args.reward_only_positive=True \
#     seed=$i \
#     save_model=False save_model_interval=200000 \
#     >/dev/null 2>&1 &
# done
# do
#     CUDA_VISIBLE_DEVICES=$cuda_device nohup python3 src/main.py \
#     --config=qmix --env-config=sc2 --map_name=$map \
#     with use_tensorboard=True env_args.reward_only_positive=False \
#     seed=$i t_max=5050000 epsilon_anneal_time=1000000 \
#     save_model=True save_model_interval=200000 \
#     >/dev/null 2>&1 &
# done