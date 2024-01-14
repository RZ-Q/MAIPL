import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy.special import comb
# from components.episode_buffer import ReplayBuffer
# from components.preference import ScriptGlobalRewardModelLabelss
# from episode_buffer import EpisodeBatch, ReplayBuffer

# TODO: -mean + mean, grad clip, remove activate, optimizer change
class RewardNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, active):
        super(RewardNet, self).__init__()
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.ac1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.ac2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.ac3 = nn.LeakyReLU()
        self.out_layer = nn.Linear(hidden_size, out_size) 
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.active = active

    def forward(self, x):
        # remove active function
        x = self.ac1(self.layer1(x))
        x = self.ac2(self.layer2(x))
        x = self.ac3(self.layer3(x))
        if self.active == "sig":
            x = self.sig(self.out_layer(x))
        elif self.active == "tan":
            x = self.tan(self.out_layer(x))
        elif self.active == "no":
            x = self.out_layer(x)
        return x

# Global and Local reward model
class RewardModel:
    def __init__(self, args, mac, logger):
        # args and mac
        self.args = args
        self.mac = copy.deepcopy(mac)
        self.logger = logger

        # log
        self.total_feedback = 0.0
        self.labeled_feedback = 0
        self.acc = 0.0

        # basic infos
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.n_actions = self.args.n_actions  # reward model input onehot actions
        self.n_agents = self.args.n_agents
        self.input_action_shape = self.n_actions if self.args.actions_onehot else self.args.action_shape
        # self.act_dim = 1

        # reward model config both global and local
        self.hidden_size = self.args.reward_hidden_size
        self.ensemble_size = self.args.ensemble_size
        # global
        self.ensemble = []
        self.param_list = []
        self.lr = self.args.reward_lr
        self.optimizer = None
        # local
        self.local_ensemble = []
        self.local_param_list = []
        self.local_lr = self.args.reward_lr
        self.local_optimizer = None

        self.state_or_obs = self.args.state_or_obs
        self.actions_onehot = self.args.actions_onehot
        self.reward_update = self.args.reward_update
        self.train_batch_size = self.args.reward_train_batch_size
        self.active = self.args.active
        self.construct_ensemble()
        if self.args.loss_func == "cross_entropy":
            self.loss_func = self.cross_entropy_loss 
        elif self.args.loss_func == "KL":
            self.loss_func = self.KL_loss

        # sample config
        self.sample_episode_size = self.args.sample_episode_size
        self.sample_segment_size = self.args.sample_segment_size
        self.origin_sample_segment_size = self.args.sample_segment_size
        self.segment_size = self.args.segment_size

        # segment pairs buffer config
        self.segment_capacity = self.args.segment_capacity
        self.buffer = {
            "state_segment1": [],
            "state_segment2": [],
            "obs_segment1": [],
            "obs_segment2": [],
            "label": [],
            "local_label": [],
            "local_label1": [],
            "local_label2": [],
            "mask1": [],
            "mask2": [],
        }
        self.buffer_index = 0
        self.buffer_full = False

        # label stuff
        self.global_preference_type = self.args.global_preference_type
        self.local_preference_type = self.args.local_preference_type
        if self.global_preference_type == "policy" or self.local_preference_type== "policy":
            self.mac.load_models(self.args.policy_dir)
            if self.args.use_cuda:
                self.mac.cuda()
    
    def get_feedbacks(self):
        return self.total_feedback
    
    def get_acc(self):
        return self.acc

    def change_batch(self, new_frac):
        self.sample_segment_size = int(new_frac * self.origin_sample_segment_size)
    
    def construct_ensemble(self):
        for _ in range(self.ensemble_size):
            model = (
                RewardNet(
                    in_size=self.state_shape + self.input_action_shape * self.n_agents
                    if self.state_or_obs
                    else self.n_agents * (self.input_action_shape + self.obs_shape + self.n_agents),
                    out_size=1,
                    hidden_size=self.hidden_size,
                    active=self.active
                )
                .float()
                .to(self.args.device)
            )
            self.ensemble.append(model)
            self.param_list.extend(model.parameters())
            local_model = (
                RewardNet(
                    in_size=self.input_action_shape + self.obs_shape + self.n_agents,  # n_agnets for agent id
                    out_size=1,
                    hidden_size=self.hidden_size,
                    active=self.active
                )
                .float()
                .to(self.args.device)
            )
            self.local_ensemble.append(local_model)
            self.local_param_list.extend(local_model.parameters())
        self.optimizer = torch.optim.Adam(self.param_list, lr=self.lr)
        self.local_optimizer = torch.optim.Adam(self.local_param_list, lr=self.local_lr)
    
    def uniform_sampling(self, buffer):
        # get queries
        query1, query2 = self.get_queries(buffer)

        # get labels
        # global_labels shape (mb_size, 2)
        gloabl_labels, local_labels = self.get_labels(query1, query2)

        if len(gloabl_labels) > 0:
            self.put_queries(query1, query2, gloabl_labels, local_labels)

        return len(gloabl_labels)

    def get_queries(self, buffer):
        # sample latest episodes from replay buffer
        sampled_episodes = buffer.sample_latest(self.sample_episode_size)
        num_episodes = sampled_episodes.batch_size
        # get mask
        terminated = sampled_episodes["terminated"].float()
        mask = sampled_episodes["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        sampled_episodes["mask"] = mask

        # sample index
        batch_index_1 = np.random.choice(num_episodes, size=self.sample_segment_size, replace=True)
        batch_index_2 = np.random.choice(num_episodes, size=self.sample_segment_size, replace=True)

        # init queries
        query1 = {
            "state": [],
            "obs": [],
            "actions": [],
            "avail_actions": [],
            "actions_onehot": [],
            "reward": [],
            "indi_reward": [],
            "mask": [],
        }
        query2 = copy.deepcopy(query1)

        for i in range(self.sample_segment_size):
            index1 = int(batch_index_1[i])
            index2 = int(batch_index_2[i])
            # true length
            len1 = int(mask[index1].sum())
            len2 = int(mask[index2].sum())

            # prevent from out of range
            if len1 > self.segment_size:
                time_index_1 = np.random.choice(len1 - self.segment_size)
                for key in query1.keys():
                    query1[key].append(sampled_episodes[index1][key][:, time_index_1 : time_index_1 + self.segment_size])
            else:
                for key in query1.keys():
                    query1[key].append(sampled_episodes[index1][key][:, :self.segment_size])
            if len2 > self.segment_size:
                time_index_2 = np.random.choice(len2 - self.segment_size)
                for key in query2.keys():
                    query2[key].append(sampled_episodes[index2][key][:, time_index_2 : time_index_2 + self.segment_size])
            else:
                for key in query2.keys():
                    query2[key].append(sampled_episodes[index2][key][:, :self.segment_size])

        return query1, query2

    def get_labels(self, query1, query2):
        global_labels = self.get_global_labels(query1, query2)
        if self.args.use_local_reward:
            local_labels = self.get_local_labels_between_seg(query1, query2)
        else:
            local_labels = torch.zeros((global_labels.shape[0], self.n_agents, 2))
        # local_labels = self.get_local_labels(query1, query2)
        # local_labels = self.get_local_labels_k_wise(query1, query2)
        return global_labels, local_labels
    
    def get_global_labels(self, query1, query2):
        if self.global_preference_type == "true_rewards":
            r_1 = torch.cat(query1["reward"]).sum(1)
            r_2 = torch.cat(query2["reward"]).sum(1)
            label = 1.0 * (r_1 < r_2)
            return torch.cat((1-label, label),dim=-1)
        elif self.global_preference_type == "policy":
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
            cum_p1 = torch.prod(torch.prod(chosen_p1 * mask1 + (1 - mask1), dim=1) * 1e2, dim=-1)
            cum_p2 = torch.prod(torch.prod(chosen_p2 * mask2 + (1 - mask2), dim=1) * 1e2, dim=-1)
            label = 1.0 * (cum_p1 < cum_p2)
            return torch.stack((1-label, label),dim=-1)

    def get_local_labels(self, query1, query2):
        agent_num = self.args.n_agents
        if self.local_preference_type == "true_indi_rewards":
            r_1 = torch.cat(query1["indi_reward"], dim=0).sum(1)
            r_2 = torch.cat(query2["indi_reward"], dim=0).sum(1)
            delta_r1 = []
            delta_r2 = []
            for i in range(agent_num):
                for j in range(i + 1, agent_num):
                    delta_r1.append((r_1[:, i] - r_1[:, j]).squeeze(-1))
                    delta_r2.append((r_2[:, i] - r_2[:, j]).squeeze(-1))
            delta_r1 = torch.cat(delta_r1, dim=-1)
            delta_r2 = torch.cat(delta_r2, dim=-1)   
            thres1 = torch.sort(delta_r1.abs()).values[int(self.args.lcoal_label_equal_thres * delta_r1.shape[0])]
            thres2 = torch.sort(delta_r2.abs()).values[int(self.args.lcoal_label_equal_thres * delta_r2.shape[0])]
            # thres1 = -1
            # thres2 = -1
            labels1, labels2 = [], []
            for i in range(agent_num):
                for j in range(i + 1, agent_num):
                    label1 = 0.5 * (torch.abs(r_1[:, i] - r_1[:, j]) <= thres1)
                    label1 += 1.0 * ((torch.abs(r_1[:, i] - r_1[:, j]) > thres1) & (r_1[:, i] < r_1[:, j]))
                    labels1.append(torch.stack((1-label1, label1), dim=-1))
                    label2 = 0.5 * (torch.abs(r_2[:, i] - r_2[:, j]) <= thres2)
                    label2 += 1.0 * ((torch.abs(r_2[:, i] - r_2[:, j]) > thres2) & (r_2[:, i] < r_2[:, j]))
                    labels2.append(torch.stack((1-label2, label2), dim=-1))
            return torch.stack(labels1, dim=1), torch.stack(labels2, dim=1)
        elif self.local_preference_type == "policy":
            mac_out1, mac_out2 = [], []
            with torch.no_grad(): # reference   
                self.mac.init_hidden(len(query1["state"]))    
                for t in range(query1["state"][0].shape[1]): 
                    agent_outs1 = self.mac.forward_query(query1, t=t)
                    agent_outs2 = self.mac.forward_query(query2, t=t)
                    mac_out1.append(agent_outs1)
                    mac_out2.append(agent_outs2)
            
            mac_out1 = torch.stack(mac_out1, dim=1)  # Concat over time
            mac_out2 = torch.stack(mac_out2, dim=1)
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
            cum_p1 = torch.prod(chosen_p1 * mask1 + (1 - mask1), dim=1) * 1e2
            cum_p2 = torch.prod(chosen_p2 * mask2 + (1 - mask2), dim=1) * 1e2

            labels1, labels2 = [], []
            # TODO: add threshould
            for i in range(agent_num):
                for j in range(i + 1, agent_num):
                    label1 = 1.0 * (cum_p1[:, i] < cum_p1[:, j])
                    labels1.append(torch.stack((1-label1, label1), dim=-1))
                    label2 = 1.0 * (cum_p2[:, i] < cum_p2[:, j])
                    labels2.append(torch.stack((1-label2, label2), dim=-1))
            return torch.stack(labels1, dim=1), torch.stack(labels2, dim=1)

    def get_local_labels_between_seg(self, query1, query2):
        if self.local_preference_type == "true_indi_rewards":
            r_1 = torch.cat(query1["indi_reward"], dim=0).sum(1)
            r_2 = torch.cat(query2["indi_reward"], dim=0).sum(1)
            labels = 1.0 * (r_1 < r_2)
            return torch.stack((1-labels, labels),dim=-1)
        elif self.local_preference_type == "policy":
            mac_out1, mac_out2 = [], []
            with torch.no_grad(): # reference   
                self.mac.init_hidden(len(query1["state"]))    
                for t in range(query1["state"][0].shape[1]): 
                    agent_outs1 = self.mac.forward_query(query1, t=t)
                    agent_outs2 = self.mac.forward_query(query2, t=t)
                    mac_out1.append(agent_outs1)
                    mac_out2.append(agent_outs2)
            
            mac_out1 = torch.stack(mac_out1, dim=1)  # Concat over time
            mac_out2 = torch.stack(mac_out2, dim=1)
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
            cum_p1 = torch.prod(chosen_p1 * mask1 + (1 - mask1), dim=1) * 1e2
            cum_p2 = torch.prod(chosen_p2 * mask2 + (1 - mask2), dim=1) * 1e2
            labels = 1.0 * (cum_p1 < cum_p2)
            return torch.stack((1-labels, labels),dim=-1)

    def get_local_labels_k_wise(self, query1, query2):
        if self.local_preference_type == "true_indi_rewards":
            agent_num = self.args.n_agents
            labels1, labels2 = [], []
            r_1 = torch.cat(query1["indi_reward"], dim=0).sum(1)
            r_2 = torch.cat(query2["indi_reward"], dim=0).sum(1)
            return torch.sort(r_1, dim=-1, descending=True)[1], torch.sort(r_2, dim=-1, descending=True)[1]
        elif self.local_preference_type == "policy":
            mac_out1, mac_out2 = [], []
            with torch.no_grad(): # reference   
                self.mac.init_hidden(len(query1["state"]))    
                for t in range(query1["state"][0].shape[1]): 
                    agent_outs1 = self.mac.forward_query(query1, t=t)
                    agent_outs2 = self.mac.forward_query(query2, t=t)
                    mac_out1.append(agent_outs1)
                    mac_out2.append(agent_outs2)
            
            mac_out1 = torch.stack(mac_out1, dim=1)  # Concat over time
            mac_out2 = torch.stack(mac_out2, dim=1)
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
            cum_p1 = torch.prod(torch.prod(chosen_p1 * mask1 + (1 - mask1), dim=1) * 1e2, dim=-1)
            cum_p2 = torch.prod(torch.prod(chosen_p2 * mask2 + (1 - mask2), dim=1) * 1e2, dim=-1)

            labels1, labels2 = [], []
            for i in range(agent_num):
                for j in range(i + 1, agent_num):
                    label1 = 1.0 * (cum_p1[:, i] < cum_p1[:, j])
                    labels1.append(torch.stack((1-label1, label1), dim=-1))
                    label2 = 1.0 * (cum_p2[:, i] < cum_p2[:, j])
                    labels2.append(torch.stack((1-label2, label2), dim=-1))
            return torch.stack(labels1, dim=1), torch.stack(labels2, dim=1)

    def put_queries(self, query1, query2, labels, local_labels):
        total_samples = len(query1["state"])
        ids = torch.eye(self.n_agents).expand(total_samples, self.segment_size, -1, -1)
        next_index = self.buffer_index + total_samples
        if next_index >= self.segment_capacity:
            self.buffer_full = True
            max_index = self.segment_capacity - self.buffer_index
            if self.actions_onehot:
                state_segment1 = torch.cat((torch.cat(query1["state"][:max_index], dim=0),
                    torch.cat(query1["actions_onehot"][:max_index], dim=0).view(
                        max_index, self.segment_size, -1)), dim=-1)
                state_segment2 = torch.cat((torch.cat(query2["state"][:max_index], dim=0),
                    torch.cat(query2["actions_onehot"][:max_index], dim=0).view(
                        max_index, self.segment_size, -1)), dim=-1)
                obs_segment1 = torch.cat((torch.cat(query1["obs"][:max_index], dim=0), ids[:max_index],
                    torch.cat(query1["actions_onehot"][:max_index], dim=0)), dim=-1)
                obs_segment2 = torch.cat((torch.cat(query2["obs"][:max_index], dim=0), ids[:max_index],
                    torch.cat(query2["actions_onehot"][:max_index], dim=0)), dim=-1)
            else:
                state_segment1 = torch.cat((torch.cat(query1["state"][:max_index], dim=0),
                    torch.cat(query1["actions"][:max_index], dim=0).squeeze(-1)), dim=-1)
                state_segment2 = torch.cat((torch.cat(query2["state"][:max_index], dim=0),
                    torch.cat(query2["actions"][:max_index], dim=0).squeeze(-1)), dim=-1)
                obs_segment1 = torch.cat((torch.cat(query1["obs"][:max_index], dim=0), ids[:max_index],
                    torch.cat(query1["actions"][:max_index], dim=0)), dim=-1)
                obs_segment2 = torch.cat((torch.cat(query2["obs"][:max_index], dim=0), ids[:max_index],
                    torch.cat(query2["actions"][:max_index], dim=0)), dim=-1)
            self.buffer["state_segment1"][self.buffer_index : self.segment_capacity] = state_segment1
            self.buffer["state_segment2"][self.buffer_index : self.segment_capacity] = state_segment2
            self.buffer["obs_segment1"][self.buffer_index : self.segment_capacity] = obs_segment1
            self.buffer["obs_segment2"][self.buffer_index : self.segment_capacity] = obs_segment2
            self.buffer["label"][self.buffer_index : self.segment_capacity] = labels[:max_index]
            # self.buffer["local_label1"][self.buffer_index : self.segment_capacity] = local_labels[0][:max_index]
            # self.buffer["local_label2"][self.buffer_index : self.segment_capacity] = local_labels[1][:max_index]
            self.buffer["local_label"][self.buffer_index : self.segment_capacity] = local_labels[:max_index]
            self.buffer["mask1"][self.buffer_index : self.segment_capacity] = query1["mask"][:max_index]
            self.buffer["mask2"][self.buffer_index : self.segment_capacity] = query2["mask"][:max_index]

            remain = total_samples - max_index
            if remain > 0:
                if self.actions_onehot:
                    state_segment1 = torch.cat((torch.cat(query1["state"][max_index:], dim=0),
                        torch.cat(query1["actions_onehot"][max_index:], dim=0).view(
                            remain, self.segment_size, -1)), dim=-1)
                    state_segment2 = torch.cat((torch.cat(query2["state"][max_index:], dim=0),
                        torch.cat(query2["actions_onehot"][max_index:], dim=0).view(
                            remain, self.segment_size, -1)), dim=-1)
                    obs_segment1 = torch.cat((torch.cat(query1["obs"][max_index:], dim=0), ids[max_index:],
                        torch.cat(query1["actions_onehot"][max_index:], dim=0)), dim=-1)
                    obs_segment2 = torch.cat((torch.cat(query2["obs"][max_index:], dim=0), ids[max_index:],
                        torch.cat(query2["actions_onehot"][max_index:], dim=0)), dim=-1)
                else:
                    state_segment1 = torch.cat((torch.cat(query1["state"][max_index:], dim=0),
                        torch.cat(query1["actions"][max_index:], dim=0).squeeze(-1)), dim=-1)
                    state_segment2 = torch.cat((torch.cat(query2["state"][max_index:], dim=0),
                        torch.cat(query2["actions"][max_index:], dim=0).squeeze(-1)), dim=-1)
                    obs_segment1 = torch.cat((torch.cat(query1["obs"][max_index:], dim=0), ids[max_index:],
                        torch.cat(query1["actions"][max_index:], dim=0)), dim=-1)
                    obs_segment2 = torch.cat((torch.cat(query2["obs"][max_index:], dim=0), ids[max_index:],
                        torch.cat(query2["actions"][max_index:], dim=0)), dim=-1)
                self.buffer["state_segment1"][:remain] = state_segment1
                self.buffer["state_segment2"][:remain] = state_segment2
                self.buffer["obs_segment1"][:remain] = obs_segment1
                self.buffer["obs_segment2"][:remain] = obs_segment2
                self.buffer["label"][:remain] = labels[max_index:]
                # self.buffer["local_label1"][:remain] = local_labels[0][max_index:]
                # self.buffer["local_label2"][:remain] = local_labels[1][max_index:]
                self.buffer["local_label"][:remain] = local_labels[max_index:]
                self.buffer["mask1"][:remain] = query1["mask"][max_index:]
                self.buffer["mask2"][:remain] = query2["mask"][max_index:]
            self.buffer_index = remain
        else:
            if self.actions_onehot:
                state_segment1 = torch.cat((torch.cat(query1["state"][:], dim=0),
                    torch.cat(query1["actions_onehot"][:], dim=0).view(
                        total_samples, self.segment_size, -1)), dim=-1)
                state_segment2 = torch.cat((torch.cat(query2["state"][:], dim=0),
                    torch.cat(query2["actions_onehot"][:], dim=0).view(
                        total_samples, self.segment_size, -1)), dim=-1)
                obs_segment1 = torch.cat((torch.cat(query1["obs"][:], dim=0), ids[:],
                    torch.cat(query1["actions_onehot"][:], dim=0)), dim=-1)
                obs_segment2 = torch.cat((torch.cat(query2["obs"][:], dim=0), ids[:],
                    torch.cat(query2["actions_onehot"][:], dim=0)), dim=-1)
            else:
                state_segment1 = torch.cat((torch.cat(query1["state"][:], dim=0),
                    torch.cat(query1["actions"][:], dim=0).squeeze(-1)), dim=-1)
                state_segment2 = torch.cat((torch.cat(query2["state"][:], dim=0),
                    torch.cat(query2["actions"][:], dim=0).squeeze(-1)), dim=-1)
                obs_segment1 = torch.cat((torch.cat(query1["obs"][:], dim=0), ids[:],
                    torch.cat(query1["actions"][:], dim=0)), dim=-1)
                obs_segment2 = torch.cat((torch.cat(query2["obs"][:], dim=0), ids[:],
                    torch.cat(query2["actions"][:], dim=0)), dim=-1)
            self.buffer["state_segment1"][self.buffer_index : next_index] = state_segment1
            self.buffer["state_segment2"][self.buffer_index : next_index] = state_segment2
            self.buffer["obs_segment1"][self.buffer_index : next_index] = obs_segment1
            self.buffer["obs_segment2"][self.buffer_index : next_index] = obs_segment2
            self.buffer["label"][self.buffer_index : next_index] = labels[:]
            # self.buffer["local_label1"][self.buffer_index : next_index] = local_labels[0][:]
            # self.buffer["local_label2"][self.buffer_index : next_index] = local_labels[1][:]
            self.buffer["local_label"][self.buffer_index : next_index] = local_labels[:]
            self.buffer["mask1"][self.buffer_index : next_index] = query1["mask"][:]
            self.buffer["mask2"][self.buffer_index : next_index] = query2["mask"][:]
            
            self.buffer_index = next_index
        
    def r_hat_member(self, x, member=-1):
        # x to cuda
        return self.ensemble[member](x.float().to(self.args.device))
    
    def local_r_hat_member(self, x, member=-1):
        # x to cuda
        return self.local_ensemble[member](x.float().to(self.args.device))

    # generate r_hat for replay buffer
    def r_hat(self, x):
        # (bs, len, dim)
        # only reference
        with torch.no_grad():
            r_hats = []
            for member in range(self.ensemble_size):
                r_hats.append(self.r_hat_member(x, member=member))
        return sum(r_hats) / len(r_hats)
    
    def local_r_hat(self, x):
        with torch.no_grad():
            r_hats = []
            for member in range(self.ensemble_size):
                r_hats.append(self.local_r_hat_member(x, member=member))
        return sum(r_hats) / len(r_hats)

    def cross_entropy_loss(self, r_hat, labels):
        # r_hat shape (mb_size, 2), labels shape (mb_size, 2)
        return torch.mean(-torch.sum(torch.mul(torch.log(torch.softmax(r_hat, dim=-1) + 1e-6), labels), dim=-1))

    def KL_loss(self, r_hat, labels):
        # r_hat shape (mb_size, 2), labels shape (mb_size, 2)
        return torch.sum(labels * ((labels + 1e-6).log() - (torch.softmax(r_hat, dim=-1) + 1e-6).log()), dim=-1).mean()


    def train_global_reward(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        max_len = self.segment_capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            loss = 0.0
            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            for member in range(self.ensemble_size):
                # get random batch
                idxs = total_batch_index[member][
                    epoch * self.train_batch_size : last_index
                ]
                traj_t_1, traj_t_2, labels = [], [], []
                mask1, mask2 = [], []
                for idx in idxs:
                    if self.state_or_obs:
                        traj_t_1.append(self.buffer["state_segment1"][idx])
                        traj_t_2.append(self.buffer["state_segment2"][idx])
                    else:
                        traj_t_1.append(self.buffer["obs_segment1"][idx].view(self.segment_size, -1))
                        traj_t_2.append(self.buffer["obs_segment2"][idx].view(self.segment_size, -1))
                    labels.append(self.buffer["label"][idx])
                    mask1.append(self.buffer["mask1"][idx])
                    mask2.append(self.buffer["mask2"][idx])
                # cat inputs (bs, seg_size, dim), (bs,2)
                traj_t_1 = torch.stack(traj_t_1, dim=0)
                traj_t_2 = torch.stack(traj_t_2, dim=0)   
                labels = torch.stack(labels, dim=0)
                mask1 = torch.cat(mask1, dim=0).to(self.args.device)
                mask2 = torch.cat(mask2, dim=0).to(self.args.device)
                if member == 0:
                    total += labels.size(0)

                # r_hat shape (bs, seg_size, 1), should be masked then sum
                r_hat1 = (self.r_hat_member(traj_t_1, member=member) * mask1).sum(1)
                r_hat2 = (self.r_hat_member(traj_t_2, member=member) * mask2).sum(1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                labels = labels.to(self.args.device)

                # compute loss
                curr_loss = self.loss_func(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                _, labels = torch.max(labels, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            
            loss.backward()
            self.optimizer.step()
        ensemble_acc = ensemble_acc / total
        return ensemble_acc

    def train_local_reward(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0.0 for _ in range(self.ensemble_size)])

        max_len = self.segment_capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.local_optimizer.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            for member in range(self.ensemble_size):
                # get random batch
                idxs = total_batch_index[member][
                    epoch * self.train_batch_size : last_index
                ]
                traj_t_1, traj_t_2, labels1, labels2 = [], [], [], []
                mask1, mask2 = [], []
                for idx in idxs:
                    traj_t_1.append(self.buffer["obs_segment1"][idx])
                    traj_t_2.append(self.buffer["obs_segment2"][idx])
                    labels1.append(self.buffer["local_label1"][idx])
                    labels2.append(self.buffer["local_label2"][idx])
                    mask1.append(self.buffer["mask1"][idx])
                    mask2.append(self.buffer["mask2"][idx])
                # cat inputs (bs, seg_size, dim), (bs,2)
                traj_t_1 = torch.stack(traj_t_1, dim=0)
                traj_t_2 = torch.stack(traj_t_2, dim=0)   
                labels1 = torch.stack(labels1, dim=0)
                labels2 = torch.stack(labels2, dim=0)
                mask1 = torch.cat(mask1, dim=0).to(self.args.device)
                mask2 = torch.cat(mask2, dim=0).to(self.args.device)

                total = 2 * labels1.size(0) * labels1.size(1)

                # r_hat shape (bs, num_agent, 1), labels shape (bs, C_n^2, 2)
                r_hat1 = (self.local_r_hat_member(traj_t_1, member=member).squeeze(-1) * mask1).sum(1).unsqueeze(-1)
                r_hat2 = (self.local_r_hat_member(traj_t_2, member=member).squeeze(-1) * mask2).sum(1).unsqueeze(-1)
                labels1 = labels1.to(self.args.device)
                labels2 = labels2.to(self.args.device)
                label_mask1 = 1.0 * (labels1[:, :, 0] != 0.5).unsqueeze(-1)
                label_mask2 = 1.0 * (labels2[:, :, 0] != 0.5).unsqueeze(-1)
                labels1 = labels1 * label_mask1
                labels2 = labels2 * label_mask2
                label_mask = torch.cat([label_mask1, label_mask2], dim=0)
                
                # compute loss
                curr_loss = []
                for i in range(self.n_agents):
                    for j in range(i + 1, self.n_agents):
                        r_i_j = torch.cat([torch.cat([r_hat1[:, i], r_hat1[:, j]], dim=-1),\
                                            torch.cat([r_hat2[:, i], r_hat2[:, j]], dim=-1)], dim=0)
                        labels = torch.cat([labels1[:, i+j-1], labels2[:, i+j-1]], dim=0)
                        loss_i_j = self.loss_func(r_i_j, labels)
                        curr_loss.append(loss_i_j)
                curr_loss = sum(curr_loss) / len(curr_loss)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                r_hat = []
                for i in range(self.n_agents):
                    for j in range(i + 1, self.n_agents):
                        r_i_j = torch.cat([torch.cat([r_hat1.data[:, i], r_hat1.data[:, j]], dim=-1),\
                                            torch.cat([r_hat2.data[:, i], r_hat2.data[:, j]], dim=-1)], dim=0)
                        r_hat.append(r_i_j)
                r_hat = torch.stack(r_hat, dim=1)
                # predicted = 0.5 * (r_hat[:, :, 0] == r_hat[:, :, 1])
                predicted = 1.0 * (r_hat[:, :, 0] < r_hat[:, :, 1]) 
                predicted = torch.stack((1-predicted, predicted), dim=-1) * label_mask
                labels = torch.cat([labels1, labels2], dim=0) * label_mask
                correct = (predicted== labels)[:, :, 0].sum().item() + label_mask.sum().item() - total
                ensemble_acc[member] += correct/(label_mask.sum().item())
                # predicted = torch.stack((1-predicted, predicted), dim=-1)
                # labels = torch.cat([labels1, labels2], dim=0)
                # correct = (predicted== labels)[:, :, 0].sum().item()
                # ensemble_acc[member] += correct/total

            loss.backward()
            self.local_optimizer.step()

        return ensemble_acc / num_epochs

    def train_local_reward_between_seg(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0.0 for _ in range(self.ensemble_size)])

        max_len = self.segment_capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))

        total = 0
        for epoch in range(num_epochs):
            self.local_optimizer.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            for member in range(self.ensemble_size):
                # get random batch
                idxs = total_batch_index[member][
                    epoch * self.train_batch_size : last_index
                ]
                traj_t_1, traj_t_2, labels = [], [], []
                mask1, mask2 = [], []
                for idx in idxs:
                    traj_t_1.append(self.buffer["obs_segment1"][idx])
                    traj_t_2.append(self.buffer["obs_segment2"][idx])
                    labels.append(self.buffer["local_label"][idx])
                    mask1.append(self.buffer["mask1"][idx])
                    mask2.append(self.buffer["mask2"][idx])
                # cat inputs (bs, seg_size, dim), (bs,2)
                traj_t_1 = torch.stack(traj_t_1, dim=0)
                traj_t_2 = torch.stack(traj_t_2, dim=0)
                labels = torch.stack(labels, dim=0)
                mask1 = torch.cat(mask1, dim=0).to(self.args.device)
                mask2 = torch.cat(mask2, dim=0).to(self.args.device)

                if member == 0:
                    total += labels.size(0) * labels.size(1)

                # r_hat shape (bs, num_agent, 1), labels shape (bs, C_n^2, 2)
                r_hat1 = (self.local_r_hat_member(traj_t_1, member=member).squeeze(-1) * mask1).sum(1).unsqueeze(-1)
                r_hat2 = (self.local_r_hat_member(traj_t_2, member=member).squeeze(-1) * mask2).sum(1).unsqueeze(-1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                labels = labels.to(self.args.device)
                
                # compute loss
                curr_loss = self.loss_func(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, -1)
                _, labels = torch.max(labels, -1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.local_optimizer.step()
        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def train_local_reward_k_wise(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        max_len = self.segment_capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.local_optimizer.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            for member in range(self.ensemble_size):
                # get random batch
                idxs = total_batch_index[member][
                    epoch * self.train_batch_size : last_index
                ]
                traj_t_1, traj_t_2, labels1, labels2 = [], [], [], []
                mask1, mask2 = [], []
                for idx in idxs:
                    traj_t_1.append(self.buffer["obs_segment1"][idx].view(self.segment_size, -1))
                    traj_t_2.append(self.buffer["obs_segment2"][idx].view(self.segment_size, -1))
                    labels1.append(self.buffer["local_label1"][idx])
                    labels2.append(self.buffer["local_label2"][idx])
                    mask1.append(self.buffer["mask1"][idx])
                    mask2.append(self.buffer["mask2"][idx])
                # cat inputs (bs, seg_size, dim), (bs,2)
                traj_t_1 = torch.stack(traj_t_1, dim=0)
                traj_t_2 = torch.stack(traj_t_2, dim=0)   
                labels1 = torch.stack(labels1, dim=0)
                labels2 = torch.stack(labels2, dim=0)
                mask1 = torch.cat(mask1, dim=0).to(self.args.device)
                mask2 = torch.cat(mask2, dim=0).to(self.args.device)

                if member == 0:
                    total += 2 * labels1.size(0) * labels1.size(1)

                # r_hat shape (bs, num_agent, 1), labels shape (bs, C_n^2, 2)
                r_hat1 = (self.local_r_hat_member(traj_t_1, member=member) * mask1).sum(1).unsqueeze(-1)
                r_hat2 = (self.local_r_hat_member(traj_t_2, member=member) * mask2).sum(1).unsqueeze(-1)
                labels1 = labels1.to(self.args.device)
                labels2 = labels2.to(self.args.device)
                
                # compute loss
                curr_loss = []
                for i in range(labels1.shape[0]):
                    r1 = r_hat1[i][labels1[i]].squeeze(-1)
                    r2 = r_hat2[i][labels2[i]].squeeze(-1)
                    p1 = p2 = 0.
                    for j in range(self.n_agents):
                        p1 += -torch.log(torch.softmax(r1[j:], dim=-1)[0] + 1e-6)
                        p2 += -torch.log(torch.softmax(r2[j:], dim=-1)[0] + 1e-6)
                    curr_loss.append(p1 + p2)
                curr_loss = sum(curr_loss) / len(curr_loss)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                r_hat = torch.cat((r_hat1.squeeze(-1), r_hat2.squeeze(-1)), dim=0)
                predicted = torch.sort(r_hat, dim=-1)[1]
                labels = torch.cat([labels1, labels2], dim=0)
                correct = (labels == predicted).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.local_optimizer.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc


    def learn_reward(self, buffer):
        labeled_queries = self.uniform_sampling(buffer)
        self.total_feedback += self.sample_segment_size
        self.labeled_feedback += labeled_queries

        if self.args.use_global_reward:
            train_acc = 0
            if self.labeled_feedback > 0:
                # update reward
                for _ in range(self.reward_update):
                    train_acc = self.train_global_reward()
                    total_acc = np.mean(train_acc)
                    if total_acc > 0.97:
                        break
            self.logger.console_logger.info(
                "Global Reward function is updated!! ACC:{}".format(str(total_acc)))
        
        if self.args.use_local_reward:
            train_acc = 0
            if self.labeled_feedback > 0:
                # update reward
                for _ in range(self.args.reward_update):
                    train_acc = self.train_local_reward_between_seg()
                    total_acc = np.mean(train_acc)
                    if total_acc > self.args.local_acc:
                        print("update:", _)
                        break
            self.acc = total_acc
            self.logger.console_logger.info(
                "Local Reward function is updated!! ACC:{}".format(str(total_acc)))



# not use anymore
class GlobalRewardModel:
    def __init__(self, args, mac):
        # args and mac
        self.args = args
        self.mac = copy.deepcopy(mac)

        # log
        self.total_feedback = 0.0
        self.labeled_feedback = 0
        self.acc = 0.0

        # basic infos
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.n_actions = self.args.n_actions  # reward model input onehot actions
        self.n_agents = self.args.n_agents
        self.input_action_shape = self.n_actions if self.args.actions_onehot else self.args.action_shape
        # self.act_dim = 1

        # reward model config
        self.hidden_size = self.args.reward_hidden_size
        self.ensemble_size = self.args.ensemble_size
        self.ensemble = []
        self.param_list = []
        self.lr = self.args.reward_lr
        self.optimizer = None
        self.state_or_obs = self.args.state_or_obs
        self.actions_onehot = self.args.actions_onehot
        self.reward_update = self.args.reward_update
        self.train_batch_size = self.args.reward_train_batch_size
        self.active = self.args.active
        self.construct_ensemble()
        if self.args.loss_func == "cross_entropy":
            self.loss_func = self.cross_entropy_loss 
        elif self.args.loss_func == "KL":
            self.loss_func = self.KL_loss

        # sample config
        self.sample_episode_size = self.args.sample_episode_size
        self.sample_segment_size = self.args.sample_segment_size
        self.origin_sample_segment_size = self.args.sample_segment_size
        self.segment_size = self.args.segment_size

        # segment pairs buffer config
        self.segment_capacity = self.args.segment_capacity
        self.buffer = {
            "state_segment1": [],
            "state_segment2": [],
            "obs_segment1": [],
            "obs_segment2": [],
            "label": [],
            "mask1": [],
            "mask2": [],
        }
        self.buffer_index = 0
        self.buffer_full = False

        # label stuff
        self.global_preference_type = self.args.global_preference_type
        if self.global_preference_type  == "policy":
            self.mac.load_models(self.args.policy_dir)
            if self.args.use_cuda:
                self.mac.cuda()
    
    def change_batch(self, new_frac):
        self.sample_segment_size = int(new_frac * self.origin_sample_segment_size)
    
    def construct_ensemble(self):
        for _ in range(self.ensemble_size):
            model = (
                RewardNet(
                    in_size=self.state_shape + self.input_action_shape * self.n_agents
                    if self.state_or_obs
                    else self.n_agents * (self.input_action_shape + self.obs_shape + self.n_agents),
                    out_size=1,
                    hidden_size=self.hidden_size,
                    active=self.active
                )
                .float()
                .to(self.args.device)
            )
            self.ensemble.append(model)
            self.param_list.extend(model.parameters())
        self.optimizer = torch.optim.Adam(self.param_list, lr=self.lr)
    
    def uniform_sampling(self, buffer):
        # get queries
        query1, query2 = self.get_queries(buffer)

        # get labels
        # global_labels shape (mb_size, 2)
        gloabl_labels = self.get_labels(query1, query2)

        if len(gloabl_labels) > 0:
            self.put_queries(query1, query2, gloabl_labels)

        return len(gloabl_labels)

    def get_queries(self, buffer):
        # sample latest episodes from replay buffer
        sampled_episodes = buffer.sample_latest(self.sample_episode_size)
        num_episodes = sampled_episodes.batch_size
        # get mask
        terminated = sampled_episodes["terminated"].float()
        mask = sampled_episodes["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        sampled_episodes["mask"] = mask

        # sample index
        batch_index_1 = np.random.choice(num_episodes, size=self.sample_segment_size, replace=True)
        batch_index_2 = np.random.choice(num_episodes, size=self.sample_segment_size, replace=True)

        # init queries
        query1 = {
            "state": [],
            "obs": [],
            "actions": [],
            "avail_actions": [],
            "actions_onehot": [],
            "reward": [],
            "mask": [],
        }
        query2 = copy.deepcopy(query1)

        for i in range(self.sample_segment_size):
            index1 = int(batch_index_1[i])
            index2 = int(batch_index_2[i])
            # true length
            len1 = int(mask[index1].sum())
            len2 = int(mask[index2].sum())

            # prevent from out of range
            if len1 > self.segment_size:
                time_index_1 = np.random.choice(len1 - self.segment_size)
                for key in query1.keys():
                    query1[key].append(sampled_episodes[index1][key][:, time_index_1 : time_index_1 + self.segment_size])
            else:
                for key in query1.keys():
                    query1[key].append(sampled_episodes[index1][key][:, :self.segment_size])
            if len2 > self.segment_size:
                time_index_2 = np.random.choice(len2 - self.segment_size)
                for key in query2.keys():
                    query2[key].append(sampled_episodes[index2][key][:, time_index_2 : time_index_2 + self.segment_size])
            else:
                for key in query2.keys():
                    query2[key].append(sampled_episodes[index2][key][:, :self.segment_size])

        return query1, query2

    def get_labels(self, query1, query2):
        if self.global_preference_type == "true_rewards":
            r_1 = torch.cat(query1["reward"]).sum(1)
            r_2 = torch.cat(query2["reward"]).sum(1)
            label = 1.0 * (r_1 < r_2)
            return torch.cat((1-label, label),dim=-1)
        elif self.global_preference_type == "policy":
            mac_out1, mac_out2 = [], []
            with torch.no_grad(): # reference 
                self.mac.init_hidden(len(query1["state"]))
                for t in range(query1["state"][0].shape[1]): 
                    agent_outs1 = self.mac.forward_query(query1, t=t)
                    agent_outs2 = self.mac.forward_query(query2, t=t)
                    mac_out1.append(agent_outs1)
                    mac_out2.append(agent_outs2)
            
            mac_out1 = torch.stack(mac_out1, dim=1)  # Concat over time
            mac_out2 = torch.stack(mac_out2, dim=1)
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
            cum_p1 = torch.prod(torch.prod(chosen_p1 * mask1 + (1 - mask1), dim=1) * 1e2, dim=-1)
            cum_p2 = torch.prod(torch.prod(chosen_p2 * mask2 + (1 - mask2), dim=1) * 1e2, dim=-1)
            label = 1.0 * (cum_p1 < cum_p2)
            return torch.stack((1-label, label),dim=-1)
    
    def put_queries(self, query1, query2, labels):
        total_samples = len(query1["state"])
        next_index = self.buffer_index + total_samples
        ids = torch.eye(self.n_agents).expand(total_samples, self.segment_size, -1, -1)
        #TODO: add ids
        if next_index >= self.segment_capacity:
            self.buffer_full = True
            max_index = self.segment_capacity - self.buffer_index
            if self.actions_onehot:
                state_segment1 = torch.cat((torch.cat(query1["state"][:max_index], dim=0),
                    torch.cat(query1["actions_onehot"][:max_index], dim=0).view(
                        max_index, self.segment_size, -1)), dim=-1)
                state_segment2 = torch.cat((torch.cat(query2["state"][:max_index], dim=0),
                    torch.cat(query2["actions_onehot"][:max_index], dim=0).view(
                        max_index, self.segment_size, -1)), dim=-1)
                obs_segment1 = torch.cat((torch.cat(query1["obs"][:max_index], dim=0),
                    torch.cat(query1["actions_onehot"][:max_index], dim=0)), dim=-1)
                obs_segment2 = torch.cat((torch.cat(query2["obs"][:max_index], dim=0),
                    torch.cat(query2["actions_onehot"][:max_index], dim=0)), dim=-1)
            else:
                state_segment1 = torch.cat((torch.cat(query1["state"][:max_index], dim=0),
                    torch.cat(query1["actions"][:max_index], dim=0).squeeze(-1)), dim=-1)
                state_segment2 = torch.cat((torch.cat(query2["state"][:max_index], dim=0),
                    torch.cat(query2["actions"][:max_index], dim=0).squeeze(-1)), dim=-1)
                obs_segment1 = torch.cat((torch.cat(query1["obs"][:max_index], dim=0),
                    torch.cat(query1["actions"][:max_index], dim=0)), dim=-1)
                obs_segment2 = torch.cat((torch.cat(query2["obs"][:max_index], dim=0),
                    torch.cat(query2["actions"][:max_index], dim=0)), dim=-1)
            self.buffer["state_segment1"][self.buffer_index : self.segment_capacity] = state_segment1
            self.buffer["state_segment2"][self.buffer_index : self.segment_capacity] = state_segment2
            self.buffer["obs_segment1"][self.buffer_index : self.segment_capacity] = obs_segment1
            self.buffer["obs_segment2"][self.buffer_index : self.segment_capacity] = obs_segment2
            self.buffer["label"][self.buffer_index : self.segment_capacity] = labels[:max_index]
            self.buffer["mask1"][self.buffer_index : self.segment_capacity] = query1["mask"][:max_index]
            self.buffer["mask2"][self.buffer_index : self.segment_capacity] = query2["mask"][:max_index]

            remain = total_samples - max_index
            if remain > 0:
                if self.actions_onehot:
                    state_segment1 = torch.cat((torch.cat(query1["state"][max_index:], dim=0),
                        torch.cat(query1["actions_onehot"][max_index:], dim=0).view(
                            remain, self.segment_size, -1)), dim=-1)
                    state_segment2 = torch.cat((torch.cat(query2["state"][max_index:], dim=0),
                        torch.cat(query2["actions_onehot"][max_index:], dim=0).view(
                            remain, self.segment_size, -1)), dim=-1)
                    obs_segment1 = torch.cat((torch.cat(query1["obs"][max_index:], dim=0),
                        torch.cat(query1["actions_onehot"][max_index:], dim=0)), dim=-1)
                    obs_segment2 = torch.cat((torch.cat(query2["obs"][max_index:], dim=0),
                        torch.cat(query2["actions_onehot"][max_index:], dim=0)), dim=-1)
                else:
                    state_segment1 = torch.cat((torch.cat(query1["state"][max_index:], dim=0),
                        torch.cat(query1["actions"][max_index:], dim=0).squeeze(-1)), dim=-1)
                    state_segment2 = torch.cat((torch.cat(query2["state"][max_index:], dim=0),
                        torch.cat(query2["actions"][max_index:], dim=0).squeeze(-1)), dim=-1)
                    obs_segment1 = torch.cat((torch.cat(query1["obs"][max_index:], dim=0),
                        torch.cat(query1["actions"][max_index:], dim=0)), dim=-1)
                    obs_segment2 = torch.cat((torch.cat(query2["obs"][max_index:], dim=0),
                        torch.cat(query2["actions"][max_index:], dim=0)), dim=-1)
                self.buffer["state_segment1"][:remain] = state_segment1
                self.buffer["state_segment2"][:remain] = state_segment2
                self.buffer["obs_segment1"][:remain] = obs_segment1
                self.buffer["obs_segment2"][:remain] = obs_segment2
                self.buffer["label"][:remain] = labels[max_index:]
                self.buffer["mask1"][:remain] = query1["mask"][max_index:]
                self.buffer["mask2"][:remain] = query2["mask"][max_index:]
            self.buffer_index = remain
        else:
            if self.actions_onehot:
                state_segment1 = torch.cat((torch.cat(query1["state"][:], dim=0),
                    torch.cat(query1["actions_onehot"][:], dim=0).view(
                        total_samples, self.segment_size, -1)), dim=-1)
                state_segment2 = torch.cat((torch.cat(query2["state"][:], dim=0),
                    torch.cat(query2["actions_onehot"][:], dim=0).view(
                        total_samples, self.segment_size, -1)), dim=-1)
                obs_segment1 = torch.cat((torch.cat(query1["obs"][:], dim=0),
                    torch.cat(query1["actions_onehot"][:], dim=0)), dim=-1)
                obs_segment2 = torch.cat((torch.cat(query2["obs"][:], dim=0),
                    torch.cat(query2["actions_onehot"][:], dim=0)), dim=-1)
            else:
                state_segment1 = torch.cat((torch.cat(query1["state"][:], dim=0),
                    torch.cat(query1["actions"][:], dim=0).squeeze(-1)), dim=-1)
                state_segment2 = torch.cat((torch.cat(query2["state"][:], dim=0),
                    torch.cat(query2["actions"][:], dim=0).squeeze(-1)), dim=-1)
                obs_segment1 = torch.cat((torch.cat(query1["obs"][:], dim=0),
                    torch.cat(query1["actions"][:], dim=0)), dim=-1)
                obs_segment2 = torch.cat((torch.cat(query2["obs"][:], dim=0),
                    torch.cat(query2["actions"][:], dim=0)), dim=-1)
            self.buffer["state_segment1"][self.buffer_index : next_index] = state_segment1
            self.buffer["state_segment2"][self.buffer_index : next_index] = state_segment2
            self.buffer["obs_segment1"][self.buffer_index : next_index] = obs_segment1
            self.buffer["obs_segment2"][self.buffer_index : next_index] = obs_segment2
            self.buffer["label"][self.buffer_index : next_index] = labels[:]
            self.buffer["mask1"][self.buffer_index : next_index] = query1["mask"][:]
            self.buffer["mask2"][self.buffer_index : next_index] = query2["mask"][:]
            
            self.buffer_index = next_index
        
    def r_hat_member(self, x, member=-1):
        # x to cuda
        return self.ensemble[member](x.float().to(self.args.device))
    
    # generate r_hat for replay buffer
    def r_hat(self, x):
        # (bs, len, dim)
        # only reference
        with torch.no_grad():
            r_hats = []
            for member in range(self.ensemble_size):
                r_hats.append(self.r_hat_member(x, member=member))
        return sum(r_hats) / len(r_hats)

    def cross_entropy_loss(self, r_hat, labels):
        # r_hat shape (mb_size, 2), labels shape (mb_size, 2)
        return torch.mean(-torch.sum(torch.mul(torch.log(torch.softmax(r_hat, dim=-1) + 1e-6), labels), dim=-1))

    def KL_loss(self, r_hat, labels):
        # r_hat shape (mb_size, 2), labels shape (mb_size, 2)
        return torch.sum(labels * ((labels + 1e-6).log() - (torch.softmax(r_hat, dim=-1) + 1e-6).log()), dim=-1).mean()


    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        max_len = self.segment_capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            loss = 0.0
            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            for member in range(self.ensemble_size):
                # get random batch
                idxs = total_batch_index[member][
                    epoch * self.train_batch_size : last_index
                ]
                traj_t_1, traj_t_2, labels = [], [], []
                mask1, mask2 = [], []
                for idx in idxs:
                    if self.state_or_obs:
                        traj_t_1.append(self.buffer["state_segment1"][idx])
                        traj_t_2.append(self.buffer["state_segment2"][idx])
                    else:
                        traj_t_1.append(self.buffer["obs_segment1"][idx].view(self.segment_size, -1))
                        traj_t_2.append(self.buffer["obs_segment2"][idx].view(self.segment_size, -1))
                    labels.append(self.buffer["label"][idx])
                    mask1.append(self.buffer["mask1"][idx])
                    mask2.append(self.buffer["mask2"][idx])
                # cat inputs (bs, seg_size, dim), (bs,2)
                traj_t_1 = torch.stack(traj_t_1, dim=0)
                traj_t_2 = torch.stack(traj_t_2, dim=0)   
                labels = torch.stack(labels, dim=0)
                mask1 = torch.cat(mask1, dim=0).to(self.args.device)
                mask2 = torch.cat(mask2, dim=0).to(self.args.device)
                if member == 0:
                    total += labels.size(0)

                # r_hat shape (bs, seg_size, 1), should be masked then sum
                r_hat1 = (self.r_hat_member(traj_t_1, member=member) * mask1).sum(1)
                r_hat2 = (self.r_hat_member(traj_t_2, member=member) * mask2).sum(1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                labels = labels.to(self.args.device)

                # compute loss
                curr_loss = self.loss_func(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                _, labels = torch.max(labels, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            
            loss.backward()
            self.optimizer.step()
        ensemble_acc = ensemble_acc / total
        return ensemble_acc


    def learn_reward(self, buffer):
        labeled_queries = self.uniform_sampling(buffer)
        self.total_feedback += self.sample_segment_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for _ in range(self.reward_update):
                train_acc = self.train_reward()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break

        print("Global Reward function is updated!! ACC: " + str(total_acc))
 