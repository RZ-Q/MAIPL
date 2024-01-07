import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np


class DMAQ_qattenLearnerGlobalRM:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.target_mixer = copy.deepcopy(self.mixer)

        self.mixer_params = list(self.mixer.parameters())
        self.q_params = list(mac.parameters())
        self.mixer_optimizer = RMSprop(params=self.mixer_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.q_optimizer = RMSprop(params=self.q_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        # Get the relevant quantities
        rewards = batch["reward_hat"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
            # self.logger.log_stat('agent_1_%d_q_1' % save_data[0], np.squeeze(q_data)[0], t_env)
            # self.logger.log_stat('agent_2_%d_q_2' % save_data[1], np.squeeze(q_data)[1], t_env)

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
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        self.mixer_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params + self.q_params, self.args.grad_norm_clip)
        self.mixer_optimizer.step()
        self.q_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.sub_train(batch, t_env, episode_num, show_demo=show_demo, save_data=save_data)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def unsup_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, full_batch: EpisodeBatch, show_demo=False, save_data=None):
        # state entropy rewards stuff
        # Get the relevant quantities
        # get batch mask
        # rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # get full batch mask
        full_terminated = full_batch["terminated"][:, :-1].float()
        full_mask = full_batch["filled"][:, :-1].float()
        full_mask[:, 1:] = full_mask[:, 1:] * (1 - full_terminated[:, :-1])

        # Compute state entropy rewards
        full_state = full_batch["state"][:, :-1]
        state = batch["state"][:, :-1]

        rewards = self._compute_state_entropy_rewards(state, full_state, mask, full_mask)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
            # self.logger.log_stat('agent_1_%d_q_1' % save_data[0], np.squeeze(q_data)[0], t_env)
            # self.logger.log_stat('agent_2_%d_q_2' % save_data[1], np.squeeze(q_data)[1], t_env)

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
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        self.mixer_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params + self.q_params, self.args.grad_norm_clip)
        self.mixer_optimizer.step()
        self.q_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _compute_state_entropy_rewards(self, state, full_state, mask, full_mask):
        # state_entropy rewards based on state
        full_state = full_state.to(self.args.device)
        full_mask = full_mask.to(self.args.device)
        (
            batch_size,
            seq_len,
            state_dim,
        ) = state.shape  # state shape (batch_size, seq_len, state_dim)
        full_size, full_state_len, _ = full_state.shape
        # add mask
        mask = mask.squeeze(-1)
        full_mask = full_mask.squeeze(-1)
        # TODO: 100 -> inf instead
        state[mask == 0] = 100
        full_state[full_mask == 0] = 100
        state = state.contiguous().view(-1, state_dim)
        full_state = full_state.contiguous().view(-1, state_dim)
        dists = []
        for idx in range(full_size // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = th.norm(
                (state[:, None, :] - full_state[None, start:end, :]), dim=-1, p=2
            )
            dists.append(dist)
        dists = th.cat(dists, dim=-1)
        knn_dists = th.kthvalue(dists, k=self.args.k, dim=1).values
        state_entropy = knn_dists.view(batch_size, seq_len).unsqueeze(-1)

        return state_entropy

    def reset_critic(self):
        if self.mixer is not None:
            if self.args.mixer == "dmaq":
                self.mixer = DMAQer(self.args)
            elif self.argsargs.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(self.args)
            else:
                raise ValueError("Mixer {} not recognised.".format(self.args.mixer))
            self.target_mixer = copy.deepcopy(self.mixer)
            # reset to cuda is use_cuda
            if self.args.use_cuda:
                self.mixer.cuda()
                self.target_mixer.cuda()

            self.mixer_params = list(self.mixer.parameters())
            self.mixer_optimizer = RMSprop(params=self.mixer_params, lr=self.args.lr, \
                                           alpha=self.args.optim_alpha, eps=self.args.optim_eps)

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
        th.save(self.mixer_optimizer.state_dict(), "{}/mixeropt.th".format(path))
        th.save(self.q_optimizer.state_dict(), "{}/qopt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.mixer_optimizer.load_state_dict(th.load("{}/mixeropt.th".format(path), map_location=lambda storage, loc: storage))
        self.q_optimizer.load_state_dict(th.load("{}/qopt.th".format(path), map_location=lambda storage, loc: storage))