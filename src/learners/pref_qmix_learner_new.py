import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class Pref_QMIX_learner_new:
    def __init__(self, mac, scheme, logger, args, pref_mac):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.pref_mac = pref_mac
        self.pref_mac.load_models(self.args.policy_dir)
        if self.args.use_cuda:
            self.pref_mac.cuda()

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.target_mixer = copy.deepcopy(self.mixer)

        # seperate mixer and agent network
        self.mixer_params = list(self.mixer.parameters())
        self.q_params = list(mac.parameters())
        self.mixer_optimizer = RMSprop(params=self.mixer_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.q_optimizer = RMSprop(params=self.q_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, reward_model):
        # Get the relevant quantities
        """
        rewards shape [bs, ep_len, 1]
        actions shape [bs, ep_len, num_agent, 1]
        avail_actions shape [bs, ep_len, num_agent, num_action]
        terminated shape [bs, ep_len, 1]
        indi_terminated shape [bs, ep_len, num_agents]
        mac_out shape [bs, ep_len+1, num_agent, num_action]
        chosen_action_qvals before mixer shape [bs, ep_len, num_agents]
        target_mac_out shape [bs, ep_len, num_agent, num_action]
        """
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        # add individual terminated info
        indi_terminated = batch["indi_terminated"][:, :-1].float()

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            # clone Qi and target_Qi_max
            chosen_action_qvals_clone = chosen_action_qvals.clone().detach()
            chosen_action_qvals_clone.requires_grad = True 
            target_max_qvals_clone = target_max_qvals.clone().detach()
            # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            chosen_action_q_tot_vals = self.mixer(chosen_action_qvals_clone, batch["state"][:, :-1])
            target_max_q_tot_vals = self.target_mixer(target_max_qvals_clone, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_q_tot_vals # [bs, ep_len, 1]

        # Td-error
        td_error = (chosen_action_q_tot_vals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        mixer_loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.mixer_optimizer.zero_grad()
        # Enables .grad attribute for non-leaf Tensors
        # 对于非叶子节点的 tensor 计算其grad(或者启用用于保存梯度的属性.grad)
        chosen_action_qvals_clone.retain_grad() #the grad of qi
        chosen_action_q_tot_vals.retain_grad() #the grad of qtot
        mixer_loss.backward()
        
        grad_l_qtot = chosen_action_q_tot_vals.grad.repeat(1, 1, self.args.n_agents) + 1e-8
        grad_l_qi = chosen_action_qvals_clone.grad
        grad_qtot_qi = th.clamp(grad_l_qi/ grad_l_qtot, min=-10, max=10) #(B,T,n_agents)

        mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)
        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.mixer_optimizer.step()
        q_rewards = self.cal_indi_reward(grad_qtot_qi, td_error.clone().detach(), chosen_action_qvals, target_max_qvals, indi_terminated) #(B,T,n_agents)
        q_rewards_clone = q_rewards.clone().detach()

        # Calculate 1-step Q-Learning targets
        q_targets = q_rewards_clone + self.args.gamma * (1 - indi_terminated) * target_max_qvals #(B,T,n_agents)

        # Td-error
        q_td_error = (chosen_action_qvals - q_targets.detach()) #(B,T,n_agents)

        
        q_mask = batch["filled"][:, :-1].float().repeat(1, 1, self.args.n_agents) #(B,T,n_agents)
        q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1]) * (1 - terminated[:, :-1]).repeat(1, 1, self.args.n_agents)
        # q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1])
        q_mask = q_mask.expand_as(q_td_error)

        masked_q_td_error = q_td_error * q_mask 

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_q_td_error ** 2).sum() / q_mask.sum()

        # preference loss
        preference_labels, preference_mask = self.get_local_labels(batch)
        preference_loss = self.cal_preference_loss(q_rewards, preference_mask, preference_labels)

        # lamda decay
        if self.args.lamda_decay:
            lamda = self.args.lamda * (self.args.t_max - t_env) / self.args.t_max
        else:
            lamda = self.args.lamda

        if self.args.direct_local_preference:
            if t_env > self.args.apply_local_timesteps:
                q_total_loss = q_loss / (q_loss.abs().detach() + 1e-6) + lamda * preference_loss / (preference_loss.abs().detach() + 1e-6)
            else:
                q_total_loss = q_loss
        else:
            if self.args.use_local_reward:
                if t_env > self.args.apply_local_timesteps:
                    q_total_loss = q_loss / (q_loss.abs().detach() + 1e-6) + lamda * preference_loss / (preference_loss.abs().detach() + 1e-6)
                else:
                    q_total_loss = q_loss
            else:
                q_total_loss = q_loss

        # Optimize
        self.q_optimizer.zero_grad()
        q_total_loss.backward()
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("mixer_loss", mixer_loss.item(), t_env)
            self.logger.log_stat("mixer_grad_norm", mixer_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("mixer_td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("mixer_target_mean", (targets * mask).sum().item()/mask_elems, t_env)

            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("q_grad_norm", q_grad_norm, t_env)
            self.logger.log_stat("prefer_loss", preference_loss.item(), t_env)
            q_mask_elems = q_mask.sum().item()
            self.logger.log_stat("q_td_error_abs", (masked_q_td_error.abs().sum().item()/q_mask_elems), t_env)
            self.logger.log_stat("q_q_taken_mean", (chosen_action_qvals * q_mask).sum().item()/(q_mask_elems), t_env)
            self.logger.log_stat("q_q_target_mean", (q_targets * q_mask).sum().item()/(q_mask_elems), t_env)
            self.logger.log_stat("reward_i_mean", (q_rewards * q_mask).sum().item()/(q_mask_elems), t_env)
            self.logger.log_stat("q_return_mean", (q_rewards * q_mask).sum(1).mean().item(), t_env)

            self.log_stats_t = t_env

    def get_local_labels(self, batch):
        # get mask
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:,:-1].float()
        mask = batch["filled"][:,:-1].float()
        indi_terminated = batch["indi_terminated"][:,:-1].float()
        indi_mask = 1 - indi_terminated
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        indi_mask[:, 1:] = batch["filled"][:,:-1].float()[:, 1:] * (1 - indi_terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        with th.no_grad(): # reference        
            self.pref_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs =  self.pref_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_out[avail_actions == 0] = -1e10
        mac_out = th.softmax(mac_out, dim=-1)
        chosen_logp = (th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) + 1e-6).log()

        labels = []
        indi_masks = []
        for i in range(self.args.n_agents):
            for j in range(i + 1, self.args.n_agents):
                indi_mask_ij = indi_mask[:,:,i] * indi_mask[:,:,j]
                label = 1.0 * ((chosen_logp[:,:,i] * indi_mask_ij).sum(1) < (chosen_logp[:,:,j] * indi_mask_ij).sum(1))
                labels.append(th.stack((1-label, label), dim=-1))
                indi_masks.append(indi_mask_ij)
        return th.stack(labels, dim=1), th.stack(indi_masks, dim=-1)

    # def cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals):
    def cal_indi_reward(self, grad_qtot_qi, mixer_td_error, qi, target_qi, indi_terminated):
        # input: grad_qtot_qi (B,T,n_agents)  mixer_td_error (B,T,1)  qi (B,T,n_agents)  indi_terminated (B,T,n_agents)
        grad_td = th.mul(grad_qtot_qi, mixer_td_error.repeat(1, 1, self.args.n_agents)) #(B,T,n_agents)
        reward_i = - grad_td + qi - self.args.gamma * (1 - indi_terminated) * target_qi
        return reward_i
    
    def cal_preference_loss(self, q_rewards, preference_mask, preference_labels):
        # q_rewards [bs, seq_len, n_agent]
        # preference_labels [bs, C_n_agent^2, 2]
        # use log(softmax) instead of nn.CEloss for 0.5 labels
        # q_rewards = (q_rewards * q_mask).sum(dim=1) # [bs, n_agent]
        # no need for label_mask,cuz agents wont die from the begining
        preference_loss = []
        for i in range(self.args.n_agents):
            for j in range(i + 1, self.args.n_agents):
                r_i = (q_rewards[:,:,i] * preference_mask[:,:,i+j-1]).sum(-1).unsqueeze(-1)
                r_j = (q_rewards[:,:,j] * preference_mask[:,:,i+j-1]).sum(-1).unsqueeze(-1)
                r_i_j = th.cat([r_i, r_j], axis=-1)
                labels = preference_labels[:, i+j-1]
                # KL loss
                loss = th.sum(labels * ((labels + 1e-6).log() - (th.softmax(r_i_j, dim=-1) + 1e-6).log()), dim=-1).mean()
                # loss = th.mean(-th.sum(th.mul(th.log(th.softmax(r_i_j, dim=-1) + 1e-6), labels), dim=-1))
                preference_loss.append(loss)
        
        return sum(preference_loss) / len(preference_loss)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.q_optimizer.state_dict(), "{}/q_opt.th".format(path))
        th.save(self.mixer_optimizer.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.q_optimizer.load_state_dict(th.load("{}/q_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimizer.load_state_dict(th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))
