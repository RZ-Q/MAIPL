import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from modules.mixers.qmix_central_attention import QMixerCentralAtten
import torch as th
from torch.optim import RMSprop
from collections import deque
from controllers import REGISTRY as mac_REGISTRY


class MAXQLearnerGlobalRM:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.mac_optimizer = RMSprop(params=self.mac_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.last_target_update_episode = 0

        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            self.mixer_optimizer = RMSprop(params=self.mac_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # Central Q
        # TODO: Clean this mess up!
        self.central_mac = None
        if self.args.central_mixer in ["ff", "atten"]:
            if self.args.central_loss == 0:
                self.central_mixer = self.mixer
                self.central_mac = self.mac
                self.target_central_mac = self.target_mac
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(args) # Feedforward network that takes state and agent utils as input
                elif self.args.central_mixer == "atten":
                    self.central_mixer = QMixerCentralAtten(args)
                else:
                    raise Exception("Error with central_mixer")

                assert args.central_mac == "basic_central_mac"
                self.central_mac = mac_REGISTRY[args.central_mac](scheme, args) # Groups aren't used in the CentralBasicController. Little hacky
                self.target_central_mac = copy.deepcopy(self.central_mac)
                self.central_mac_params = list(self.central_mac.parameters())
                self.central_mac_optimizer = RMSprop(params=self.mac_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        else:
            raise Exception("Error with qCentral")
        self.central_mixer_params = list(self.central_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.central_mixer_optimizer = RMSprop(params=self.mac_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.grad_norm = 1
        self.mixer_norm = 1
        self.mixer_norms = deque([1], maxlen=100)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals = chosen_action_qvals_agents

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)
            target_max_agent_qvals = th.gather(target_mac_out[:,:], 3, cur_max_actions[:,:]).squeeze(3)
        else:
            raise Exception("Use double q")

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        central_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
        # Use the Qmix max actions
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)
        # ---

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals[:,1:], batch["state"][:,1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - (targets.detach()))

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1])
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

        # QMIX loss with weighting
        ws = th.ones_like(td_error) * self.args.w
        if self.args.hysteretic_qmix: # OW-QMIX
            ws = th.where(td_error < 0, th.ones_like(td_error)*1, ws) # Target is greater than current max
            w_to_use = ws.mean().item() # For logging
        else: # CW-QMIX
            is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
            qtot_larger = targets > max_action_qtot
            ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error)*1, ws) # Target is greater than current max
            w_to_use = ws.mean().item() # Average of ws for logging

        qmix_loss = (ws.detach()*(masked_td_error ** 2)).sum() / mask.sum()

        # The weightings for the different losses aren't used (they are always set to 1)
        loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss

        # Optimise
        self.mixer_optimizer.zero_grad()
        self.mac_optimizer.zero_grad()
        self.central_mixer_optimizer.zero_grad()
        self.central_mac_optimizer.zero_grad()
        loss.backward()

        # Logging
        agent_norm = 0
        for p in self.mac_params:
            param_norm = p.grad.data.norm(2)
            agent_norm += param_norm.item() ** 2
        agent_norm = agent_norm ** (1. / 2)

        mixer_norm = 0
        for p in self.mixer_params:
            param_norm = p.grad.data.norm(2)
            mixer_norm += param_norm.item() ** 2
        mixer_norm = mixer_norm ** (1. / 2)
        self.mixer_norm = mixer_norm
        self.mixer_norms.append(mixer_norm)

        grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params + self.mac_params + \
                                                self.central_mixer_params + self.central_mac_params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.mixer_optimizer.step()
        self.mac_optimizer.step()
        self.central_mixer_optimizer.step()
        self.central_mac_optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env

    def unsup_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, full_batch:EpisodeBatch):
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
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals = chosen_action_qvals_agents

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)
            target_max_agent_qvals = th.gather(target_mac_out[:,:], 3, cur_max_actions[:,:]).squeeze(3)
        else:
            raise Exception("Use double q")

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        central_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
        # Use the Qmix max actions
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)
        # ---

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals[:,1:], batch["state"][:,1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - (targets.detach()))

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1])
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

        # QMIX loss with weighting
        ws = th.ones_like(td_error) * self.args.w
        if self.args.hysteretic_qmix: # OW-QMIX
            ws = th.where(td_error < 0, th.ones_like(td_error)*1, ws) # Target is greater than current max
            w_to_use = ws.mean().item() # For logging
        else: # CW-QMIX
            is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
            qtot_larger = targets > max_action_qtot
            ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error)*1, ws) # Target is greater than current max
            w_to_use = ws.mean().item() # Average of ws for logging

        qmix_loss = (ws.detach()*(masked_td_error ** 2)).sum() / mask.sum()

        # The weightings for the different losses aren't used (they are always set to 1)
        loss = self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss

        # Optimise
        self.mixer_optimizer.zero_grad()
        self.mac_optimizer.zero_grad()
        self.central_mixer_optimizer.zero_grad()
        self.central_mac_optimizer.zero_grad()
        loss.backward()

        # Logging
        agent_norm = 0
        for p in self.mac_params:
            param_norm = p.grad.data.norm(2)
            agent_norm += param_norm.item() ** 2
        agent_norm = agent_norm ** (1. / 2)

        mixer_norm = 0
        for p in self.mixer_params:
            param_norm = p.grad.data.norm(2)
            mixer_norm += param_norm.item() ** 2
        mixer_norm = mixer_norm ** (1. / 2)
        self.mixer_norm = mixer_norm
        self.mixer_norms.append(mixer_norm)

        grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params + self.mac_params + \
                                                self.central_mixer_params + self.central_mac_params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.mixer_optimizer.step()
        self.mac_optimizer.step()
        self.central_mixer_optimizer.step()
        self.central_mac_optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            self.logger.log_stat("agent_norm", agent_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env

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
        self.mixer = None
        assert self.args.mixer is not None
        if self.args.mixer is not None:
            if self.args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif self.args.mixer == "qmix":
                self.mixer = QMixer(self.args)
            else:
                raise ValueError("Mixer {} not recognised.".format(self.args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            # reset to cuda is use_cuda
            if self.args.use_cuda:
                self.mixer.cuda()
                self.target_mixer.cuda()

            self.mixer_optimizer = RMSprop(params=self.mixer_params, lr=self.args.lr)
        
        if self.args.central_mixer in ["ff", "atten"]:
            if self.args.central_loss == 0:
                self.central_mixer = self.mixer
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(self.args) # Feedforward network that takes state and agent utils as input
                elif self.args.central_mixer == "atten":
                    self.central_mixer = QMixerCentralAtten(self.args)
                else:
                    raise Exception("Error with central_mixer")
        else:
            raise Exception("Error with qCentral")
        self.central_mixer_params = list(self.central_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.central_mixer_optimizer = RMSprop(params=self.mac_params,lr=self.args.lr, \
                                               alpha=self.args.optim_alpha, eps=self.args.optim_eps)
        # reset to cuda is use_cuda
        if self.args.use_cuda:
            self.central_mixer.cuda()
            self.target_central_mixer.cuda()


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.central_mac is not None:
            self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

    # TODO: Model saving/loading is out of date!
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.mixer_optimizer.state_dict(), "{}/mixeropt.th".format(path))
        th.save(self.mac_optimizer.state_dict(), "{}/macopt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.mixer_optimizer.load_state_dict(th.load("{}/mixeropt.th".format(path), map_location=lambda storage, loc: storage))
        self.mac_optimizer.load_state_dict(th.load("{}/qopt.th".format(path), map_location=lambda storage, loc: storage))