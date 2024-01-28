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
        self.ac1 = nn.LeakyReLU()
        self.ac2 = nn.LeakyReLU()
        self.active = active
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.ac1(self.layer1(x))
        x = self.ac2(self.layer2(x))
        if self.active == "sig":
            x = self.sig(self.layer3(x))
        elif self.active == "tan":
            x = self.tan(self.layer3(x))
        elif self.active == "no":
            x = self.layer3(x)
        else:
            x = self.relu(self.layer3(x))
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
        # global use true rewards for labeling
        # TODO: add policy labels and normlize
        query1, query2 = queries
        r_1 = torch.cat(query1["reward"]).sum(1)
        r_2 = torch.cat(query2["reward"]).sum(1)
        label = 1.0 * (r_1 < r_2)
        return torch.cat((1-label, label),dim=-1)

    def get_local_labels(self, queries):
        query1, query2 = queries
        # local reward use policy for labeling
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
        actions1 = torch.cat(query1["actions"], dim=0)
        actions2 = torch.cat(query2["actions"], dim=0)
        indi_mask1 = torch.cat(query1["indi_mask"], dim=0)
        indi_mask2 = torch.cat(query2["indi_mask"], dim=0)
        
        mac_out1[avail_actions1 == 0] = -1e10
        mac_out2[avail_actions2 == 0] = -1e10
        mac_out1 = torch.softmax(mac_out1, dim=-1)
        mac_out2 = torch.softmax(mac_out2, dim=-1)

        chosen_logp1 = (torch.gather(mac_out1, dim=3, index=actions1).squeeze(3) + 1e-6).log()
        chosen_logp2 = (torch.gather(mac_out2, dim=3, index=actions2).squeeze(3) + 1e-6).log()

        labels1, labels2 = [], []
        indi_mask1s, indi_mask2s = [], []
        # logp for norm and use min(i,j) mask
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                indi_mask1_ij = indi_mask1[:,:,i] * indi_mask1[:,:,j]
                label1 = 1.0 * ((chosen_logp1[:,:,i] * indi_mask1_ij).sum(1) < (chosen_logp1[:,:,j] * indi_mask1_ij).sum(1))
                indi_mask1s.append(indi_mask1_ij)
                labels1.append(torch.stack((1-label1, label1), dim=-1))
                indi_mask2_ij = indi_mask2[:,:,i] * indi_mask2[:,:,j]
                label2 = 1.0 * ((chosen_logp2[:,:,i] * indi_mask2_ij).sum(1) < (chosen_logp2[:,:,j] * indi_mask2_ij).sum(1))
                indi_mask2s.append(indi_mask2_ij)
                labels2.append(torch.stack((1-label2, label2), dim=-1))
        return  (torch.stack(labels1, dim=1), torch.stack(labels2, dim=1)) , (torch.stack(indi_mask1s, dim=-1), torch.stack(indi_mask2s, dim=-1))

    def global_r_hat_member(self, x, member=-1):
        return self.global_ensemble[member](x.float().to(self.args.device))

    def local_r_hat_member(self, x, member=-1):
        return self.local_ensemble[member](x.float().to(self.args.device))

    def train_global_reward(self, global_labels, queries):
        query1, query2 = queries
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        self.global_optimizer.zero_grad()
        loss = 0.0
        total = 0
        state1 = torch.cat(query1["state"], dim=0)
        state2 = torch.cat(query2["state"], dim=0)
        actions1 = torch.cat(query1["actions"], dim=0).squeeze(-1)
        actions2 = torch.cat(query2["actions"], dim=0).squeeze(-1)
        mask1 = torch.cat(query1["mask"], dim=0)
        mask2 = torch.cat(query2["mask"], dim=0)
        traj_1 = torch.cat((state1, actions1), dim=-1)
        traj_2 = torch.cat((state2, actions2), dim=-1)      
        for member in range(self.ensemble_size):
            if member == 0:
                total += global_labels.size(0)
            r_hat1 = (self.global_r_hat_member(traj_1, member=member) * mask1).sum(1)
            r_hat2 = (self.global_r_hat_member(traj_2, member=member) * mask2).sum(1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

            # compute loss
            curr_loss = self.loss_func(r_hat, global_labels)
            loss += curr_loss
            ensemble_losses[member].append(curr_loss.item())

            # compute acc
            _, predicted = torch.max(r_hat.data, 1)
            _, labels = torch.max(global_labels, 1)
            correct = (predicted == labels).sum().item()
            ensemble_acc[member] += correct
        loss.backward()
        self.global_optimizer.step()
        ensemble_acc = ensemble_acc / total
        return ensemble_acc

    def train_local_reward(self, local_labels, local_masks, queries):
        query1, query2 = queries
        local_labels1, local_labels2 = local_labels
        local_masks1, local_masks2 = local_masks
        mask = torch.cat([1.0 * (local_masks1.sum(1)!=0), 1.0 * (local_masks2.sum(1)!=0)], dim=0).unsqueeze(-1)
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])

        self.local_optimizer.zero_grad()
        loss = 0.0
        total = 0
        state1 = torch.cat(query1["agent_state"], dim=0)
        state2 = torch.cat(query2["agent_state"], dim=0)
        actions1 = torch.cat(query1["actions"], dim=0)
        actions2 = torch.cat(query2["actions"], dim=0)   
        for member in range(self.ensemble_size):
            if member == 0:
                total += mask.sum().item()
            r_hat1 = torch.gather(self.local_r_hat_member(state1, member=member), dim=-1, index=actions1).squeeze(-1)
            r_hat2 = torch.gather(self.local_r_hat_member(state2, member=member), dim=-1, index=actions2).squeeze(-1)

            # compute loss
            curr_loss = []
            r_hat = []
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    r_i = torch.cat([(r_hat1[:,:,i] * local_masks1[:,:,i+j-1]).sum(-1), (r_hat2[:,:,i] * local_masks2[:,:,i+j-1]).sum(-1)], dim=0)
                    r_j = torch.cat([(r_hat1[:,:,j] * local_masks1[:,:,i+j-1]).sum(-1), (r_hat2[:,:,j] * local_masks2[:,:,i+j-1]).sum(-1)], dim=0)
                    r_i_j = torch.stack([r_i, r_j], dim=-1)
                    labels = torch.cat([local_labels1[:, i+j-1], local_labels2[:, i+j-1]], dim=0) * mask[:,i+j-1]
                    loss_i_j = self.loss_func(r_i_j, labels)
                    curr_loss.append(loss_i_j)
                    r_hat.append(r_i_j.data)
            curr_loss = sum(curr_loss) / len(curr_loss)
            loss += curr_loss
            ensemble_losses[member].append(curr_loss.item())
            r_hat = torch.stack(r_hat, dim=1)

            # compute acc
            predicted = 1.0 * (r_hat[:, :, 0] < r_hat[:, :, 1]) 
            predicted = torch.stack((1-predicted, predicted), dim=-1) * mask
            labels = torch.cat([local_labels1, local_labels2], dim=0) * mask
            correct = (predicted == labels)[:, :, 0].sum().item() - (mask==0).sum().item()
            ensemble_acc[member] += correct

        loss.backward()
        self.local_optimizer.step()
        ensemble_acc = ensemble_acc / total
        return ensemble_acc

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
            local_labels, local_masks = self.get_local_labels(queries)
            train_acc = 0
            for _ in range(self.reward_update_times):
                train_acc = self.train_local_reward(local_labels, local_masks, queries)
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break
            self.local_acc = total_acc
            self.logger.console_logger.info(
                "Local Reward function is updated!! ACC:{}".format(str(total_acc)))
        