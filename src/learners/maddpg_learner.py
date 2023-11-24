import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam
from controllers.maddpg_controller import gumbel_softmax
from modules.critics import REGISTRY as critic_registry


class MADDPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(self.mac.parameters_by_agent())

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters_by_critic())

        self.agent_optimiser = [
            Adam(params=list(self.agent_params[i]), lr=self.args.lr)
            for i in range(len(self.agent_params))
        ]
        self.critic_optimiser = [
            Adam(params=list(self.critic_params[i]), lr=self.args.critic_lr)
            for i in range(len(self.critic_params))
        ]

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.last_target_update_episode = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions_onehot"]
        obses = batch["obs"]
        states = batch["state"]
        terminated = batch["terminated"][:, :-1].float()
        rewards = rewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        # Calculation of mask is wrong?
        # terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        # mask = 1 - terminated
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        mask = mask.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        batch_size = batch.batch_size
        max_seq_length = batch.max_seq_length

        total_critic_loss = []
        total_actor_loss = []
        total_critic_grad_norm = []
        total_agent_grad_norm = []
        total_masked_td_error = []
        total_q_taken = []
        total_targets = []
        # Update one by one
        for i in range(self.n_agents):
            # Update value / Train critic
            self.target_mac.init_hidden(batch_size)
            target_actions = []
            with th.no_grad():
                for t in range(1, max_seq_length):
                    agent_target_outs = self.target_mac.target_actions(batch, t)
                    target_actions.append(agent_target_outs)
                target_actions = th.stack(target_actions, dim=1)  # Concat over time
                target_critic_in = th.cat(
                    (obses[:, 1:], target_actions.detach()), dim=-1
                ).view(batch_size, max_seq_length - 1, -1)
                target_vals = self.target_critic(target_critic_in, i)
                target_vals = target_vals.view(batch_size, -1, 1)

                targets = (
                    rewards[:, :, i]
                    + self.args.gamma * (1 - terminated[:, :, i]) * target_vals.detach()
                ).view(-1, 1)
            total_targets.append(targets)

            critic_in = th.cat((obses[:, :-1], actions[:, :-1]), dim=-1).view(
                batch_size, max_seq_length - 1, -1
            )
            q_taken = self.critic(critic_in, i)
            total_q_taken.append(q_taken)

            # Critic loss
            td_error = q_taken.view(-1, 1) - targets.detach()
            masked_td_error = td_error * mask[:, :, i].reshape(-1, 1)
            total_masked_td_error.append(masked_td_error)
            critic_loss = (masked_td_error**2).mean()

            # Update
            self.critic_optimiser[i].zero_grad()
            critic_loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(
                list(self.critic_params[i]), self.args.grad_norm_clip
            )
            total_critic_grad_norm.append(critic_grad_norm)
            self.critic_optimiser[i].step()

            total_critic_loss.append(critic_loss)

            # Update policy / Train actor
            # TODO: test only main agent gumbel_softmax, other greedy
            self.mac.init_hidden(batch_size)
            pis = []
            joint_actions = []
            for t in range(max_seq_length - 1):
                pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1)
                joint_actions.append(gumbel_softmax(pi, hard=True))
                pis.append(pi)
            joint_actions = th.cat(joint_actions, dim=1)
            new_jiont_actions = []
            for j in range(self.n_agents):
                if i == j:
                    new_jiont_actions.append(joint_actions[:, :, j])
                else:
                    new_jiont_actions.append(joint_actions[:, :, j].detach())
            new_jiont_actions = th.cat(new_jiont_actions, dim=2).view(
                batch_size, max_seq_length - 1, self.n_agents, -1
            )

            pis = th.cat(pis, dim=1)
            agent_pi = pis[:, :, i].view(-1, self.n_actions)
            agent_pi[agent_pi <= -1e10] = 0
            critic_in = th.cat((obses[:, :-1], new_jiont_actions), dim=-1).view(
                batch_size, max_seq_length - 1, -1
            )
            q = self.critic(critic_in, i)
            q = q.reshape(-1, 1)

            # Compute the actor loss
            actor_loss = -(q * mask[:, :, i].reshape(-1, 1)).mean() + self.args.reg * (agent_pi**2).mean()

            # Optimise agents
            self.agent_optimiser[i].zero_grad()
            actor_loss.backward()
            agent_grad_norm = th.nn.utils.clip_grad_norm_(
                list(self.agent_params[i]), self.args.grad_norm_clip
            )
            total_agent_grad_norm.append(agent_grad_norm)
            self.agent_optimiser[i].step()

            total_actor_loss.append(actor_loss)

        if (
            self.args.target_update_interval_or_tau > 1
            and (episode_num - self.last_target_update_episode)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", sum(total_critic_loss).item(), t_env)
            self.logger.log_stat("pg_loss", sum(total_actor_loss).item(), t_env)
            self.logger.log_stat("critic_grad_norm", sum(total_critic_grad_norm).item(), t_env)
            self.logger.log_stat("agent_grad_norm", sum(total_agent_grad_norm).item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", sum(masked_td_error[a].abs().sum().item() for a in range(self.n_agents)) / mask_elems, t_env
            )
            self.logger.log_stat(
                "q_taken_mean", (sum(q_taken)).sum().item() / mask_elems, t_env
            )
            self.logger.log_stat(
                "target_mean", sum(targets).sum().item() / mask_elems, t_env
            )     
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        # th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        # th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        # self.agent_optimiser.load_state_dict(
        #     th.load(
        #         "{}/agent_opt.th".format(path),
        #         map_location=lambda storage, loc: storage,
        #     )
        # )
