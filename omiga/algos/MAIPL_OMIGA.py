import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.network import Actor, V_critic, Q_critic, MixNet


class MAIPL_OMIGA(object):
    def __init__(self, observation_spec, action_spec, num_agent, eval_env, config):
        self._alpha = config[
            "alpha"
        ]  # 1 for medium quality dataset of the HalfCheetah task, 10 for others
        self._gamma = config["gamma"]
        self._tau = config["tau"]
        self._hidden_sizes = config["hidden_sizes"]
        self._mix_hidden_sizes = config["mix_hidden_sizes"]
        self._batch_size = config["batch_size"]
        self._lr = config["lr"]
        self._grad_norm_clip = config["grad_norm_clip"]
        self._num_agent = num_agent
        self._device = config["device"]
        self._eval_env = eval_env
        self._iteration = 0
        self._optimizers = dict()

        # v-network
        self._v_network = V_critic(
            observation_spec, num_agent, self._hidden_sizes, self._device
        ).to(self._device)
        self._v_target_network = copy.deepcopy(self._v_network)
        self._optimizers["v"] = torch.optim.Adam(self._v_network.parameters(), self._lr)

        # q-network and mix-network
        self._q_network = Q_critic(
            observation_spec, action_spec, num_agent, self._hidden_sizes, self._device
        ).to(self._device)
        self._q_target_network = copy.deepcopy(self._q_network)
        self._mix_network = MixNet(
            observation_spec,
            action_spec,
            num_agent,
            self._mix_hidden_sizes,
            self._device,
        ).to(self._device)
        self._mix_target_network = copy.deepcopy(self._mix_network)
        self._q_param = list(self._q_network.parameters()) + list(
            self._mix_network.parameters()
        )
        self._optimizers["q"] = torch.optim.Adam(self._q_param, self._lr)

        # policy-network
        self._policy_network = Actor(
            observation_spec, action_spec, num_agent, self._hidden_sizes, self._device
        ).to(self._device)
        self._optimizers["policy"] = torch.optim.Adam(
            self._policy_network.parameters(), self._lr
        )

        # reward loss
        self._reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        self._target_clipping = config["target_clipping"]
        self._chi2_coeff = config["chi2_coeff"]
        self._chi2_replay_weight = config["chi2_replay_weight"]
        self._value_replay_weight = config["chi2_replay_weight"]
        self._policy_replay_weight = config["chi2_replay_weight"]
        # self._chi2_replay_weight = None
        # self._value_replay_weight = None
        # self._policy_replay_weight = None
        self._v_target = config["v_target"]
        self._use_pref_only = config["use_pref_only"]

    def q_loss(self, labels, o_with_a_id, s, o_next_with_id, s_next, mask, split, B_p, S_p, result={}):
        # TODO: double Q ablation (OMIGA only use one Q)
        q_values = self._q_network(o_with_a_id)
        w, b = self._mix_network(s)
        q_total = (w * q_values).sum(dim=-2) + b.squeeze(dim=-1)

        with torch.no_grad():
            if self._v_target:
                v_next_target_values = self._v_target_network(o_next_with_id)
                w_next_target, b_next_target = self._mix_target_network(s_next)
                v_next_total = (w_next_target * v_next_target_values).sum(dim=-2) + b_next_target.squeeze(dim=-1)
            else:
                v_next_values = self._v_network(o_next_with_id)
                w_next, b_next = self._mix_network(s_next)
                v_next_total = (w_next * v_next_values).sum(dim=-2) + b_next.squeeze(dim=-1)
           

            # target clipping
            if self._target_clipping:
                q_lim = 1.0 / (self._chi2_coeff * (1 - self._gamma))
                v_next_total = torch.clamp(v_next_total, min=-q_lim, max=q_lim)

        reward = q_total - self._gamma * mask * v_next_total

        # Now re-chunk everything to get the logits
        r1, r2, rr = torch.split(reward, split, dim=0)
        r1, r2 = r1.view(B_p, S_p), r2.view(B_p, S_p)
        logits = r2.sum(dim=-1) - r1.sum(dim=-1)

        # Compute the Q-loss over the imitation data
        # labels = labels.float().unsqueeze(0)
        assert labels.shape == logits.shape
        q_loss = self._reward_criterion(logits, labels).mean()

        # Compute the Chi2 Loss over EVERYTHING, including replay data
        if self._chi2_replay_weight is not None and split[-1] > 0:
            # This tries to balance the loss over data points.
            chi2_loss_fb = self._chi2_coeff * 0.5 * (torch.square(r1).mean() + torch.square(r2).mean())
            chi2_loss_replay = self._chi2_coeff * torch.square(rr).mean()
            chi2_loss = (1 - self._chi2_replay_weight) * chi2_loss_fb + self._chi2_replay_weight * chi2_loss_replay
        else:
            # default is 0.5
            # 1 / (4 * 0.5) = 1 / 2  c = 0.5 --> reward is bounded on [-2, 2]
            chi2_loss = self._chi2_coeff * (reward**2).mean()  # Otherwise compute over all

        result.update(
            {
                "q_loss": q_loss,
                "chi2_loss": chi2_loss,
                "q_total": q_total.mean(),
                "w1": w[:, 0, :].mean(),
                "w2": w[:, 1, :].mean(),
                "b": b.mean(),
                "q_values1": q_values[:, 0, :].mean(),
                "q_values2": q_values[:, 1, :].mean(),
            }
        )

        return result

    def v_loss(self, z, w_target, v_values, split, result={}):
        max_z = torch.max(z)
        max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).to(self._device), max_z)
        max_z = max_z.detach()

        # / exp(max_z) for normalize
        v_loss = (torch.exp(z - max_z) + torch.exp(-max_z) * w_target * v_values / self._alpha)

        # Allow to dynammically weight the value loss from the different sources.
        if self._value_replay_weight is not None and split[-1] > 0:
            v1, v2, vr = torch.split(v_loss, split, dim=0)
            # This tries to balance the loss over data points.
            v_loss_fb = (v1.mean() + v2.mean()) / 2
            v_loss_replay = vr.mean()
            v_loss = (1 - self._value_replay_weight) * v_loss_fb + self._value_replay_weight * v_loss_replay
        else:
            v_loss = v_loss.mean()  # Otherwise compute over all

        result.update({ "v_loss": v_loss,})
        return result

    def policy_loss(self, exp_a, a, o_with_id, split, result={}):
        log_probs = self._policy_network.get_log_density(o_with_id, a)
        policy_loss = -(exp_a * log_probs)

        if self._policy_replay_weight is not None and split[-1] > 0:
            p1, p2, pr = torch.split(policy_loss, split, dim=0)
            # This tries to balance the loss over data points.
            policy_loss_fb = (p1.mean() + p2.mean()) / 2
            policy_loss_replay = pr.mean()
            policy_loss = (1 - self._policy_replay_weight) * policy_loss_fb + self._policy_replay_weight * policy_loss_replay
        else:
            policy_loss = policy_loss.mean()  # Otherwise compute over all

        result.update({"policy_loss": policy_loss,})
        return result

    def decode_and_cat_batch(self, offline_batch=None, pref_batch=None):
        using_offline_batch = offline_batch is not None and "obs" in offline_batch

        # We need to assemble all of the observation, next observations, states, next_states and their splits to compute reward values.
        # First collect all the shapes
        B_p, S_p = pref_batch["obs1"].shape[:2]
        S_p -= 1  # Subtract one for the next_obs offset
        flat_obs_p_shape = (B_p * S_p,) + pref_batch["obs1"].shape[2:]
        flat_state_p_shape = (B_p * S_p,) + pref_batch["state1"].shape[2:]
        flat_action_p_shape = (B_p * S_p,) + pref_batch["action1"].shape[2:]
        flat_mask_p_shape = (B_p * S_p,) + pref_batch["mask1"].shape[2:]
        B_o = offline_batch["obs"].shape[0] if using_offline_batch else 0

        # Compute the split over observations between feedback and replay batches
        split = [B_p * S_p, B_p * S_p, B_o]
        B_total = split[0] + split[1] + split[2]
        # Construct one large batch for each of obs, action, next obs, state, next_state
        obs = torch.cat(
            [
                pref_batch["obs1"][:, :-1].reshape(*flat_obs_p_shape),
                pref_batch["obs2"][:, :-1].reshape(*flat_obs_p_shape),
                *((offline_batch["obs"],) if using_offline_batch else ()),
            ],
            dim=0,
        )
        state = torch.cat(
            [
                pref_batch["state1"][:, :-1].reshape(*flat_state_p_shape),
                pref_batch["state2"][:, :-1].reshape(*flat_state_p_shape),
                *((offline_batch["state"],) if using_offline_batch else ()),
            ],
            dim=0,
        )
        action = torch.cat(
            [
                pref_batch["action1"][:, :-1].reshape(*flat_action_p_shape),
                pref_batch["action2"][:, :-1].reshape(*flat_action_p_shape),
                *((offline_batch["action"],) if using_offline_batch else ()),
            ],
            dim=0,
        )
        next_obs = torch.cat(
            [
                pref_batch["obs1"][:, 1:].reshape(*flat_obs_p_shape),
                pref_batch["obs2"][:, 1:].reshape(*flat_obs_p_shape),
                *((offline_batch["obs_next"],) if using_offline_batch else ()),
            ],
            dim=0,
        )
        next_state = torch.cat(
            [
                pref_batch["state1"][:, 1:].reshape(*flat_obs_p_shape),
                pref_batch["state2"][:, 1:].reshape(*flat_obs_p_shape),
                *((offline_batch["state_next"],) if using_offline_batch else ()),
            ],
            dim=0,
        )
        mask = torch.cat(
            [
                pref_batch["mask1"][:, :-1].reshape(*flat_mask_p_shape),
                pref_batch["mask2"][:, :-1].reshape(*flat_mask_p_shape),
                *((offline_batch["mask"],) if using_offline_batch else ()),
            ],
            dim=0,
        )

        return obs, state, action, mask, next_obs, next_state, split, B_total, B_p, S_p

    def train_step(self, offline_batch=None, pref_batch=None, step=0):
        # decode and cat batch datas
        if self._use_pref_only:
            obs, state, action, mask, next_obs, next_state, split, B_total, B_p, S_p = (
                self.decode_and_cat_batch(offline_batch=None, pref_batch=pref_batch)
            )
        else:
            obs, state, action, mask, next_obs, next_state, split, B_total, B_p, S_p = (
                self.decode_and_cat_batch(offline_batch, pref_batch)
            )
        labels = pref_batch["labels"].squeeze(-1).float()

        # Shared network values
        one_hot_agent_id = (
            torch.eye(self._num_agent).expand(obs.shape[0], -1, -1).to(self._device)
        )
        o_with_id = torch.cat((obs, one_hot_agent_id), dim=-1)
        o_with_a_id = torch.cat((obs, action, one_hot_agent_id), dim=-1)
        o_next_with_id = torch.cat((next_obs, one_hot_agent_id), dim=-1)

        # q_loss
        loss_result = self.q_loss(
            labels,
            o_with_a_id,
            state,
            o_next_with_id,
            next_state,
            mask,
            split,
            B_p,
            S_p,
            result={},
        )

        # v and policy shared values
        with torch.no_grad():
            q_target_values = self._q_target_network(o_with_a_id)
            w_target, b_target = self._mix_target_network(state)

        v_values = self._v_network(o_with_id)

        z = (1 / self._alpha  * (w_target * q_target_values - w_target * v_values))
        z = torch.clamp(z, min=-10.0, max=10.0)

        exp_a = torch.exp(z).detach().squeeze(-1)

        # v_loss
        loss_result = self.v_loss(z, w_target, v_values, split, result=loss_result)
        # policy_loss
        loss_result = self.policy_loss(exp_a, action, o_with_id, split, result=loss_result)

        self._optimizers["policy"].zero_grad()
        loss_result["policy_loss"].backward()
        nn.utils.clip_grad_norm_(self._policy_network.parameters(), self._grad_norm_clip)
        self._optimizers["policy"].step()

        self._optimizers["q"].zero_grad()
        (loss_result["q_loss"] + loss_result["chi2_loss"]).backward()
        nn.utils.clip_grad_norm_(self._q_param, self._grad_norm_clip)
        self._optimizers["q"].step()

        self._optimizers["v"].zero_grad()
        loss_result["v_loss"].backward()
        nn.utils.clip_grad_norm_(self._v_network.parameters(), self._grad_norm_clip)
        self._optimizers["v"].step()

        # soft update
        for param, target_param in zip(self._q_network.parameters(), self._q_target_network.parameters()):
            target_param.data.copy_(
                self._tau * param.data + (1 - self._tau) * target_param.data
            )
        for param, target_param in zip(self._mix_network.parameters(), self._mix_target_network.parameters()):
            target_param.data.copy_(
                self._tau * param.data + (1 - self._tau) * target_param.data
            )
        for param, target_param in zip(self._v_network.parameters(), self._v_target_network.parameters()):
            target_param.data.copy_(
                self._tau * param.data + (1 - self._tau) * target_param.data
            )

        self._iteration += 1

        loss_result.update(
            {
                "v_values1": v_values[:, 0, :].mean(),
                "v_values2": v_values[:, 1, :].mean(),
                "q_target_values1": q_target_values[:, 0, :].mean(),
                "q_target_values2": q_target_values[:, 1, :].mean(),
            }
        )
        return loss_result

    def step(self, o):
        o = torch.from_numpy(o).to(self._device)
        one_hot_agent_id = (
            torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        )
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        action = self._policy_network.get_deterministic_action(o_with_id)

        return action.detach().cpu()
