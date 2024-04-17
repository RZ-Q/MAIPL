map_name="5m_vs_6m"
algo="maicpl_smac"
model_type="NMR" #model_type=$model_type train_reward=False use_reward_hat=True \ cpl_lambda=0.1 cpl_alpha=1000 \
for s in {0,1,2,3,4}
do
    CUDA_VISIBLE_DEVICES=6 python3 ICQ-MA/src/main.py \
        --config=$algo --env-config=sc2 \
        with offline_dataset_dir=/data/user/kouqian/SMAC410_pref_new/5m_vs_6m_good-medium_1000_30.th \
        env_args.map_name=$map_name offline_dataset_quality=good-medium pref_segment_pairs=1000 \
        use_wandb=True \
        cpl_lambda=0.01 cpl_alpha=1000 \
        >/dev/null 2>&1 &
done
#TODO: timesteps, dataset, segment, dense all try, reward train