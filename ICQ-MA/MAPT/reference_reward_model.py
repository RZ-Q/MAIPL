import os
import sys
import glob
import gym
import torch
import pickle
import absl.app
import absl.flags
import numpy as np
import transformers
import configparser
from collections import defaultdict
from tensorboardX import SummaryWriter
sys.path.append("../../")
from mat.algorithms.reward_model.models.MR import MR
from mat.algorithms.reward_model.models.NMR import NMR
from mat.algorithms.reward_model.models.lstm import LSTMRewardModel
from mat.algorithms.reward_model.utils.dataloader import load_dataset
from mat.algorithms.reward_model.models.PrefTransformer import PrefTransformer
from mat.algorithms.reward_model.models.trajectory_gpt2 import TransRewardModel
from mat.algorithms.reward_model.models.q_function import FullyConnectedQFunction
from mat.algorithms.reward_model.models.torch_utils import batch_to_torch, index_batch
from mat.algorithms.reward_model.models.MultiPrefTransformer import MultiPrefTransformer
from mat.algorithms.reward_model.models.encoder_decoder_divide import MultiTransRewardDivideModel
from mat.algorithms.reward_model.utils.utils import Timer, define_flags_with_default, set_random_seed, \
    get_user_flags, prefix_metrics, WandBLogger, save_pickle
     
class config:
    def __init__(self,task='',model_type='',qua=''):
        self.env='smac'
        self.task=task
        self.model_type=model_type
        self.qua=qua
        self.seed=1
        self.save_model=True
        self.batch_size=256
        self.orthogonal_init=False
        self.activations='relu'
        self.activation_final='none'
        self.n_epochs=2000
        self.eval_period=5
        self.comment='reference'
        self.max_traj_length=100
        self.multi_transformer=MultiPrefTransformer.get_default_config()
        self.transformer=PrefTransformer.get_default_config()
        self.reward=MR.get_default_config()
        self.lstm=NMR.get_default_config()
        self.logging=WandBLogger.get_default_config()
        ################ dir config
        self.device='cuda'
        self.dataset_path=""
        self.model_dir=""
        self.save_file_name=""

def main(FLAGS):
    #################### set logger dir

    FLAGS.dataset_path='/data/user/kouqian/files/MAOfflineDatasets/SMAC410_pref/'+FLAGS.task+"_"+FLAGS.qua+'_5000_32.pkl'
    FLAGS.model_dir="MAPT/results/pref_reward/smac/"+FLAGS.task+"/5000/"+FLAGS.model_type+"/"+ "_"+FLAGS.qua+"_5000_32/models/"
    FLAGS.save_file_name="/data/user/kouqian/files/MAOfflineDatasets/SMAC410_pref/"+FLAGS.task+"_"+FLAGS.qua+"_5000_32_"+FLAGS.model_type+"_preds.pkl"
    file_names = glob.glob(os.path.join(FLAGS.model_dir, "*"))
    FLAGS.model_dir = file_names[-1]


    save_file_name = FLAGS.save_file_name
    save_dir = FLAGS.logging.output_dir + '/' + FLAGS.env + '/' + FLAGS.task
    save_dir += '/' + str(FLAGS.model_type) + '/'
    save_dir += f"{FLAGS.comment}" + "/"
    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    FLAGS.logging.log_dir = save_dir + '/logs/'
    FLAGS.logging.model_dir = save_dir + '/models/'
    #################### set random seed
    set_random_seed(FLAGS.seed)
    #################################### load dataset
    action_type = 'Continous' if (FLAGS.env == 'hands' or FLAGS.env == 'mujoco') else 'Discrete'
    pref_dataset, pref_eval_dataset, env_info = load_dataset(
        FLAGS.env, FLAGS.task, FLAGS.dataset_path, action_type
    )
    data_size = pref_dataset["observations0"].shape[0]
    interval = int(data_size / FLAGS.batch_size) + 1
    print('----------------------  finish load data')
    #################################### set env info
    observation_dim, action_dim = env_info['observation_dim'], env_info['action_dim']
    FLAGS.max_traj_length = env_info['max_len']
    n_agent = env_info['n_agent']
    #################################### config reward model
    if FLAGS.model_type == "MultiPrefTransformerDivide":
        config = transformers.GPT2Config(**FLAGS.multi_transformer)
        # config multi-transformer reward model
        trans = MultiTransRewardDivideModel(
            config=config, observation_dim=observation_dim, action_dim=action_dim, n_agent=n_agent,
            action_type=action_type, max_episode_steps=FLAGS.max_traj_length, device=FLAGS.device,
        )
        # config model wrapper for train and eval
        reward_model = MultiPrefTransformer(config, trans, FLAGS.device)
    elif FLAGS.model_type == "PrefTransformer":
        config = transformers.GPT2Config(**FLAGS.transformer)
        config.warmup_steps = int(FLAGS.n_epochs * 0.1 * interval)
        config.total_steps = FLAGS.n_epochs * interval
        # config transformer reward model
        trans = TransRewardModel(
            config=config, observation_dim=observation_dim, action_dim=action_dim, action_type=action_type,
            activation=FLAGS.activations, activation_final=FLAGS.activation_final,
            max_episode_steps=FLAGS.max_traj_length, device=FLAGS.device,
        )
        # config model wrapper for train and eval
        reward_model = PrefTransformer(config, trans, FLAGS.device)
    elif FLAGS.model_type == "MR":
        rf = FullyConnectedQFunction(
            observation_dim=observation_dim, action_dim=action_dim, action_type=action_type,
            inner_dim=FLAGS.reward.inner_dim, action_embd_dim=FLAGS.reward.action_embd_dim,
            orthogonal_init=FLAGS.orthogonal_init, activations=FLAGS.activations,
            activation_final=FLAGS.activation_final, device=FLAGS.device,
        )
        reward_model = MR(FLAGS.reward, rf, FLAGS.device)
    elif FLAGS.model_type == "NMR":
        config = transformers.GPT2Config(**FLAGS.lstm)
        config.warmup_steps = int(FLAGS.n_epochs * 0.1 * interval)
        config.total_steps = FLAGS.n_epochs * interval
        lstm = LSTMRewardModel(
            config=config, observation_dim=observation_dim, action_dim=action_dim, action_type=action_type,
            activation=FLAGS.activations, activation_final=FLAGS.activation_final,
            max_episode_steps=FLAGS.max_traj_length, device=FLAGS.device,
        )
        reward_model = NMR(config, lstm, FLAGS.device)
    else:
        raise NotImplementedError()

    reward_model.load_model(model_dir=FLAGS.model_dir)
    train_dataset_pred_r0, train_dataset_pred_r1 = [], []

    #################################### run reference pipeline
    for i in range(interval):
        start_pt = i * FLAGS.batch_size
        end_pt = min((i + 1) * FLAGS.batch_size, pref_dataset["observations0"].shape[0])
        if start_pt >= end_pt:
            break
        # reference
        batch = batch_to_torch(index_batch(pref_dataset, range(start_pt, end_pt)), FLAGS.device)
        pred_r0, pred_r1 = reward_model.get_reward_for_offline(batch)
        train_dataset_pred_r0.append(pred_r0)
        train_dataset_pred_r1.append(pred_r1)
    
    pred_r0 = torch.cat(train_dataset_pred_r0, dim=0)
    pred_r1 = torch.cat(train_dataset_pred_r1, dim=0)
    preds = {
        "pred_r0": pred_r0,
        "pred_r1": pred_r1,
    }
    with open(save_file_name, "wb") as fp:
        pickle.dump(preds, fp)


if __name__ == '__main__':
    task = ['5m_vs_6m']
    quality = ['good', 'medium', 'poor', 'good_medium', 'medium_poor']
    reward_model = ['MR', 'NMR', 'MultiPrefTransformerDivide', 'PrefTransformer']
    for t in task:
        for r in reward_model:
            for q in quality:          
                FLAGS_DEF = config(
                task=t,
                model_type=r,
                qua=q,
                )
                main(FLAGS_DEF)
