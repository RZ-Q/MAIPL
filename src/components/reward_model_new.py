import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy.special import comb

# 3 layers MLP reward model
class RewardNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, active):
        super(RewardNet, self).__init__()
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_size)
        self.active = active

    def forward(self, x):
        x = nn.LeakyReLU(self.layer1(x))
        x = nn.LeakyReLU(self.layer2(x))
        if self.active == "sig":
            x = nn.Sigmoid(self.layer3(x))
        elif self.active == "tan":
            x = nn.Tanh(self.layer3(x))
        elif self.active == "no":
            x = self.layer3(x)
        else:
            x = nn.ReLU(self.out_layer(x))
        return x

class RewardModel:
    def __init__(self, args, policy_mac, logger):
        # args, policy mac and logger
        self.args = args
        self.mac = copy.deepcopy(policy_mac)  # for policy labeling
        self.logger = logger
        self.device = self.args.device

        # log infos
        self.global_labels = 0.0
        self.local_labels = 0.0
        self.global_acc = 0.0
        self.local_acc = 0.0

        # basic infos
        #TODO: look out, change to MAPPO state
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.input_action_shape = self.n_actions if self.args.actions_onehot else self.args.action_shape
        self.global_in_shape = self.args.state_shape + self.input_action_shape * self.n_agents
        self.local_in_shape = self.args.agent_state_shape  # agent_id included

        # reward model config
        self.hidden_size = self.args.reward_hidden_size
        self.ensemble_size = self.args.ensemble_size
        self.active = self.args.active
        if self.args.loss_func == "cross_entropy":
            self.loss_func = self.cross_entropy_loss 
        elif self.args.loss_func == "KL":
            self.loss_func = self.KL_loss
        self.reward_update_times = self.args.reward_update_times
        # global
        self.use_global_reward = self.args.use_global_reward
        self.global_ensemble = []
        self.global_param_list = []
        self.global_lr = self.args.reward_lr
        self.global_optimizer = None
        # local
        self.use_local_reward = self.args.use_local_reward
        self.local_ensemble = []
        self.local_param_list = []
        self.local_lr = self.args.reward_lr
        self.local_optimizer = None

        # sample config
        #TODO: add schedule, sample BS, currently same with qmix training
        self.sample_method = self.args.sample_method
        self.sample_batch_size = self.args.sample_batch_size  # same as learner currently
        self.origin_batch_size = self.args.sample_batch_size
        self.add_batch_size = self.args.add_batch_size
        self.segment_size = self.args.segment_size

        # label stuff
        # remove indi_reward preference type, only policy prefer type
        self.mac.load_models(self.args.policy_dir)
        if self.args.use_cuda:
            self.mac.cuda()
        
        # construct reward model
        self.construct_ensemble()
    
    def change_batch(self, new_frac):
        self.sample_batch_size = int(new_frac * self.origin_batch_size)

    def construct_ensemble(self):
        # add reward model to list
        for _ in range(self.ensemble_size):
            global_model = (
                RewardNet(
                    in_size=self.global_in_shape,out_size=1,
                    hidden_size=self.hidden_size,active=self.active
                )
                .float().to(self.device)
            )
            local_model = (
                RewardNet(
                    in_size=self.local_in_shape,out_size=self.n_actions,
                    hidden_size=self.hidden_size,active=self.active
                )
                .float().to(self.device)
            )
            self.global_ensemble.append(global_model)
            self.local_ensemble.append(local_model)
            self.global_param_list.extend(global_model.parameters())
            self.local_param_list.extend(local_model.parameters())
        
        self.global_optimizer = torch.optim.Adam(self.global_param_list, lr=self.global_lr)
        self.local_optimizer = torch.optim.Adam(self.local_param_list, lr=self.local_lr)
    
    def uniform_sampling(self, episode_sample):
        num_episodes = episode_sample.batch_size
        # get mask
        terminated = episode_sample["terminated"].float()
        mask = episode_sample["filled"].float()
        indi_terminated = episode_sample["indi_terminated"].float()
        indi_mask = 1 - indi_terminated
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        indi_mask[:, 1:] = episode_sample["filled"].float()[:, 1:] * (1 - indi_terminated[:, :-1])
        episode_sample["mask"] = mask
        episode_sample["indi_mask"] = indi_mask

        # sample index
        batch_index_1 = np.random.choice(num_episodes, size=self.sample_batch_size, replace=True)
        batch_index_2 = np.random.choice(num_episodes, size=self.sample_batch_size, replace=True)

        # init queries
        query1 = {
            "state": [],
            "agent_state": [],
            "obs": [],
            "actions": [],
            "avail_actions": [],
            "actions_onehot": [],
            "reward": [],
            "mask": [],
            "indi_mask": [],
        }
        query2 = copy.deepcopy(query1)

        for i in range(self.sample_batch_size):
            index1 = int(batch_index_1[i])
            index2 = int(batch_index_2[i])
            # true length
            len1 = int(mask[index1].sum())
            len2 = int(mask[index2].sum())

            # prevent from out of range
            if len1 > self.segment_size:
                time_index_1 = np.random.choice(len1 - self.segment_size)
                for key in query1.keys():
                    query1[key].append(episode_sample[index1][key][:, time_index_1 : time_index_1 + self.segment_size])
            else:
                for key in query1.keys():
                    query1[key].append(episode_sample[index1][key][:, :self.segment_size])
            if len2 > self.segment_size:
                time_index_2 = np.random.choice(len2 - self.segment_size)
                for key in query2.keys():
                    query2[key].append(episode_sample[index2][key][:, time_index_2 : time_index_2 + self.segment_size])
            else:
                for key in query2.keys():
                    query2[key].append(episode_sample[index2][key][:, :self.segment_size])

        return query1, query2

    def get_global_labels(self, queries):
        query1, query2 = queries
        mac_out1, mac_out2 = [], []
        with torch.no_grad(): # reference 
            self.mac.init_hidden(len(query1["state"]))
            for t in range(query1["state"][0].shape[1]):
                agent_outs1 = self.mac.forward_query(query1, t=t)
                agent_outs2 = self.mac.forward_query(query2, t=t)
                mac_out1.append(agent_outs1)
                mac_out2.append(agent_outs2)
        
        mac_out1 = torch.stack(mac_out1, dim=1)  # Concat over time
        mac_out2 = torch.stack(mac_out2, dim=1)  # (bs, segment_len, n_agents, n_actions)
        avail_actions1 = torch.cat(query1["avail_actions"], dim=0)
        avail_actions2 = torch.cat(query2["avail_actions"], dim=0)
        actions1 = torch.cat(query1["actions"], dim=0).to(self.args.device)
        actions2 = torch.cat(query2["actions"], dim=0).to(self.args.device)
        mask1 = torch.cat(query1["mask"], dim=0).to(self.args.device)
        mask2 = torch.cat(query2["mask"], dim=0).to(self.args.device)
        
        mac_out1[avail_actions1 == 0] = -1e10
        mac_out2[avail_actions2 == 0] = -1e10
        mac_out1 = torch.softmax(mac_out1, dim=-1)
        mac_out2 = torch.softmax(mac_out2, dim=-1)

        chosen_p1 = torch.gather(mac_out1, dim=3, index=actions1).squeeze(3)
        chosen_p2 = torch.gather(mac_out2, dim=3, index=actions2).squeeze(3)
        # multiple a small value for 0.0
        # normalize cum_p
        cum_p1 = torch.prod(torch.prod(chosen_p1 * mask1 + (1 - mask1), dim=1) * 1e2, dim=-1)
        cum_p2 = torch.prod(torch.prod(chosen_p2 * mask2 + (1 - mask2), dim=1) * 1e2, dim=-1)
        label = 1.0 * (cum_p1 < cum_p2)
        return torch.stack((1-label, label),dim=-1)

    def cross_entropy_loss(self, r_hat, labels):
        # r_hat shape (mb_size, 2), labels shape (mb_size, 2)
        return torch.mean(-torch.sum(torch.mul(torch.log(torch.softmax(r_hat, dim=-1) + 1e-6), labels), dim=-1))

    def KL_loss(self, r_hat, labels):
        # r_hat shape (mb_size, 2), labels shape (mb_size, 2)
        return torch.sum(labels * ((labels + 1e-6).log() - (torch.softmax(r_hat, dim=-1) + 1e-6).log()), dim=-1).mean()

    def learn_reward(self, episode_sample):
        # sample segment pairs for reward models
        if self.sample_method == "uniform":
            queries = self.uniform_sampling(episode_sample)
        else:
            queries = self.uniform_sampling(episode_sample)
        
        # if train rewards then get labels and train
        if self.use_global_reward:
            global_labels = self.get_global_labels(queries)
            train_acc = 0
            for _ in range(self.reward_update_times):
                train_acc = self.train_global_reward(global_labels, queries)
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break
            self.global_acc = total_acc
            self.logger.console_logger.info(
                "Global Reward function is updated!! ACC:{}".format(str(total_acc)))
        
        if self.use_local_reward:
            local_labels, local_queries = self.get_local_labels(queries)
            train_acc = 0
            for _ in range(self.reward_update_times):
                train_acc = self.train_local_reward(local_labels, local_queries)
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break
            self.local_acc = total_acc
            self.logger.console_logger.info(
                "Local Reward function is updated!! ACC:{}".format(str(total_acc)))
        