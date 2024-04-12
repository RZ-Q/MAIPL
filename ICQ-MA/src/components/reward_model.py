import torch
import numpy as np
import transformers
from tensorboardX.writer import SummaryWriter
from components.MAPT.mat.algorithms.reward_model.models.MR import MR
from components.MAPT.mat.algorithms.reward_model.models.NMR import NMR
from components.MAPT.mat.algorithms.reward_model.models.lstm import LSTMRewardModel
from components.MAPT.mat.algorithms.reward_model.models.PrefTransformer import PrefTransformer
from components.MAPT.mat.algorithms.reward_model.models.trajectory_gpt2 import TransRewardModel
from components.MAPT.mat.algorithms.reward_model.models.MultiPrefTransformer import MultiPrefTransformer
from components.MAPT.mat.algorithms.reward_model.models.encoder_decoder_divide import MultiTransRewardDivideModel
from components.MAPT.mat.algorithms.reward_model.models.q_function import FullyConnectedQFunction


class RewardModel:
    def __init__(self, args):
        self.args = args

        # configs for reward model
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.action_type = args.action_type
        self.max_traj_length = args.max_traj_length
        self.device = 'cuda' if args.use_cuda else 'cpu'
        self.model_type = args.model_type

        self.multi_transformer_config = MultiPrefTransformer.get_default_config()
        self.transformer_config = PrefTransformer.get_default_config()
        self.MR_config = MR.get_default_config()
        self.NMR_config = NMR.get_default_config()

        self.activations = 'relu'
        self.activation_final = 'none'
        self.orthogonal_init = 'False'

        self.train_epoch = args.reward_train_epoch

        if self.model_type == "MultiPrefTransformerDivide":
                config = transformers.GPT2Config(**self.multi_transformer_config)
                # config multi-transformer reward model
                trans = MultiTransRewardDivideModel(
                    config=config, observation_dim=self.obs_shape, action_dim=self.n_actions, n_agent=self.n_agents,
                    action_type=self.action_type, max_episode_steps=args.max_traj_length, device=args.device,
                )
                # config model wrapper for train and eval
                self.reward_model = MultiPrefTransformer(config, trans, self.device)
        elif self.model_type == "PrefTransformer":
            config = transformers.GPT2Config(**self.transformer_config)
            # config transformer reward model
            trans = TransRewardModel(
                config=config, observation_dim=self.obs_shape, action_dim=self.n_actions, action_type=self.action_type,
                activation=self.activations, activation_final=self.activation_final,
                max_episode_steps=self.max_traj_length, device=self.device,
            )
            # config model wrapper for train and eval
            self.reward_model = PrefTransformer(config, trans, self.device)
        elif self.model_type == "MR":
            rf = FullyConnectedQFunction(
                observation_dim=self.obs_shape, action_dim=self.n_actions, action_type=self.action_type,
                inner_dim=self.MR_config.inner_dim, action_embd_dim=self.MR_config.action_embd_dim,
                orthogonal_init=self.orthogonal_init, activations=self.activations,
                activation_final=self.activation_final, device=self.device,
            )
            self.reward_model = MR(self.MR_config, rf, self.device)
        elif self.model_type == "NMR":
            config = transformers.GPT2Config(**self.NMR_config)
            lstm = LSTMRewardModel(
                config=config, observation_dim=self.obs_shape, action_dim=self.n_actions, action_type=self.action_type,
                activation=self.activations, activation_final=self.activation_final,
                max_episode_steps=self.max_traj_length, device=self.device,
            )
            self.reward_model = NMR(config, lstm, self.device)
    
    def train(self, pref_dataset):
        train_batch = {}
        eval_batch = {}
        mask = pref_dataset['filled'].float()
        mask[:, 1:] = mask[:, 1:] * (1 - pref_dataset["terminated"][:, :-1])
        train_bs = int(self.args.reward_batch_size / 2)
        eval_bs = int(pref_dataset['obs'].shape[0] / 2)
        eval_batch['observations0'] = pref_dataset['obs'][:eval_bs, :self.max_traj_length].to(self.device)
        eval_batch['observations1'] = pref_dataset['obs'][eval_bs:, :self.max_traj_length].to(self.device)
        eval_batch['actions0'] = pref_dataset['action'][:eval_bs, :self.max_traj_length].to(self.device)
        eval_batch['actions1'] = pref_dataset['action'][eval_bs:, :self.max_traj_length].to(self.device)
        eval_batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
        eval_batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
        eval_batch['masks0'] = mask[:eval_bs, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
        eval_batch['masks1'] = mask[eval_bs:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
        eval_batch['labels'] = pref_dataset['labels'].to(self.device)
        eval_batch['labels'] = torch.cat([1 - eval_batch['labels'], eval_batch['labels']], dim=-1)

        for i in range(self.train_epoch):
            shuffled_idx = np.random.permutation(eval_bs)
            interval = int(eval_bs / train_bs) + 1
            for j in range(interval):
                start_pt0 = i * train_bs
                end_pt0 = min((i + 1) * train_bs, eval_bs)
                sample_number0 = shuffled_idx[start_pt0: end_pt0]
                sample_number1 = sample_number0 + eval_bs
                train_batch['observations0'] = pref_dataset['obs'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['actions0'] = pref_dataset['action'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['observations1'] = pref_dataset['obs'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['actions1'] = pref_dataset['action'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(train_bs, 1).to(self.device)
                train_batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(train_bs, 1).to(self.device)
                train_batch['masks0'] = mask[sample_number0][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
                train_batch['masks1'] = mask[sample_number1][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
                train_batch['labels'] = pref_dataset['labels'][sample_number0].to(self.device)
                train_batch['labels'] = torch.cat([1 - train_batch['labels'], train_batch['labels']], dim=-1)
                metrics = self.reward_model.train(train_batch)
            
            acc = self.reward_model.evaluation_acc(eval_batch)
            if acc > 0.97:
                break
        
        preds0, preds1 = self.reward_model.get_reward_for_offline(eval_batch)

        return torch.cat([preds0, preds1], dim=0)
