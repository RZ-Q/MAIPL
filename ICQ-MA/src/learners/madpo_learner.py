import copy
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
import torch as th
from utils.offpg_utils import build_target_q
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop
from modules.mixers.qmix import QMixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.time_atten import TimeattenMixer
import torch.nn.functional as F
import numpy as np
import wandb

class MADPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.dpo_lambda = args.dpo_lambda
        self.dpo_alpha = args.dpo_alpha
        self.distence_type = args.distence_type
        self.distence_log = args.distence_log
        self.agent_mixer = None
        self.time_mixer = None

        if args.agent_mixer == "attention":
            self.agent_mixer = QattenMixer(args)
        if args.time_mixer == "attention":
            self.time_mixer = TimeattenMixer(args)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.agent_params = list(mac.parameters())

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch0, batch1, t_env, labels, running_log):
        # # ----------- batch0 preferred -----------------
        states0 = batch0["state"]
        actions0 = batch0["actions"]
        rewards0 = batch0["reward"]
        terminated0 = batch0["terminated"].float()
        avail_actions0 = batch0["avail_actions"].long()
        mask0 = batch0["filled"].float()
        mask0[:, 1:] = mask0[:, 1:] * (1 - terminated0[:, :-1])


        # Calculate estimated Q-Values
        mac_out0 = []
        self.mac.init_hidden(batch0['batch_size'])
        for t in range(batch0['max_seq_length']):
            agent_outs = self.mac.forward(batch0, t=t)
            mac_out0.append(agent_outs)
        mac_out0 = th.stack(mac_out0, dim=1)

        # # ----------- batch1 not preferred -----------------
        states1 = batch1["state"]
        actions1 = batch1["actions"]
        rewards1 = batch1["reward"]
        terminated1 = batch1["terminated"].float()
        avail_actions1 = batch1["avail_actions"].long()
        mask1 = batch1["filled"].float()
        mask1[:, 1:] = mask1[:, 1:] * (1 - terminated1[:, :-1])

        # Calculate estimated Q-Values
        mac_out1 = []
        self.mac.init_hidden(batch1['batch_size'])
        for t in range(batch1['max_seq_length']):
            agent_outs = self.mac.forward(batch1, t=t)
            mac_out1.append(agent_outs)
        mac_out1 = th.stack(mac_out1, dim=1)

        mac_out0[avail_actions0 == 0] = 0
        mac_out0 = mac_out0/mac_out0.sum(dim=-1, keepdim=True)
        mac_out0[avail_actions0 == 0] = 0
        
        mac_out1[avail_actions1 == 0] = 0
        mac_out1 = mac_out1/mac_out1.sum(dim=-1, keepdim=True)
        mac_out1[avail_actions1 == 0] = 0

        if self.distence_type == "log_pi":
            pi_taken0 = th.gather(mac_out0, dim=-1, index=actions0)
            pi_taken0[mask0.repeat(1, 1, self.n_agents).unsqueeze(-1) == 0] = 1.0
            log_pi_taken0 = th.log(pi_taken0)

            pi_taken1 = th.gather(mac_out1, dim=-1, index=actions1)
            pi_taken1[mask1.repeat(1, 1, self.n_agents).unsqueeze(-1) == 0] = 1.0
            log_pi_taken1 = th.log(pi_taken1)

            if self.time_mixer == None:
                distence0 = (self.agent_mixer(-self.dpo_alpha * log_pi_taken0, states0) * mask0).sum(1) / mask0.sum(1)
                distence1 = (self.agent_mixer(-self.dpo_alpha * log_pi_taken1, states1) * mask1).sum(1) / mask1.sum(1)
            else:
                interval0 = int(rewards0.shape[1] / self.args.traj_feature_rtgs) + 1
                rtgs0 = rewards0.cumsum(dim=1).flip(dims=[1])[:, 0::interval0].squeeze(-1)
                trajs0 = th.cat([states0[:, 0], actions0[:, 0].squeeze(-1), rtgs0], dim=-1)
                distence0 = self.agent_mixer(-self.dpo_alpha * log_pi_taken0, states0)
                distence0 = self.time_mixer(distence0, states0, trajs0)
                interval1 = int(rewards1.shape[1] / self.args.traj_feature_rtgs) + 1
                rtgs1 = rewards1.cumsum(dim=1).flip(dims=[1])[:, 0::interval1].squeeze(-1)
                trajs1 = th.cat([states1[:, 0], actions1[:, 0].squeeze(-1), rtgs1], dim=-1)
                distence1 = self.agent_mixer(-self.dpo_alpha * log_pi_taken1, states1)
                distence1 = self.time_mixer(distence1, states1, trajs1)
        elif self.distence_type == "actions":
            actions1_all = th.arange(self.n_actions).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(actions1.shape[0], actions1.shape[1], actions1.shape[2], 1).to('cuda')

            distence0 = (mac_out0 * th.sqrt((actions1_all - actions0) ** 2) * avail_actions0).sum(-1)
            distence1 = (mac_out1 * th.sqrt((actions1_all - actions1) ** 2) * avail_actions1).sum(-1)

            if self.time_mixer == None:
                distence0 = (self.agent_mixer(self.dpo_alpha * distence0, states0) * mask0).sum(1) / mask0.sum(1)
                distence1 = (self.agent_mixer(self.dpo_alpha * distence1, states1) * mask1).sum(1) / mask1.sum(1)
            else:
                interval0 = int(rewards0.shape[1] / self.args.traj_feature_rtgs) + 1
                rtgs0 = rewards0.cumsum(dim=1).flip(dims=[1])[:, 0::interval0].squeeze(-1)
                trajs0 = th.cat([states0[:, 0], actions0[:, 0].squeeze(-1), rtgs0], dim=-1)
                distence0 = self.agent_mixer(self.dpo_alpha * distence0, states0)
                distence0 = self.time_mixer(distence0, states0, trajs0)
                interval1 = int(rewards1.shape[1] / self.args.traj_feature_rtgs) + 1
                rtgs1 = rewards1.cumsum(dim=1).flip(dims=[1])[:, 0::interval1].squeeze(-1)
                trajs1 = th.cat([states1[:, 0], actions1[:, 0].squeeze(-1), rtgs1], dim=-1)
                distence1 = self.agent_mixer(self.dpo_alpha * distence1, states1)
                distence1 = self.time_mixer(distence1, states1, trajs1)
        # # ------------- DPO loss ------------------
        logit10 = (-distence1) - self.dpo_lambda * (-distence0)
        logit01 = (-distence0) - self.dpo_lambda * (-distence1)
        max21 = th.clamp(-logit10, min=0, max=None)
        max12 = th.clamp(-logit01, min=0, max=None)
        nlp21 = th.log(th.exp(-max21) + th.exp(-logit10 - max21)) + max21
        nlp12 = th.log(th.exp(-max12) + th.exp(-logit01 - max12)) + max12
        loss = labels * nlp21 + (1 - labels) * nlp12
        loss = loss.mean()

        self.agent_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env
            running_log.update({
                "loss": loss.item(),
                "agent_grad_norm": grad_norm,
            })

    def cuda(self):
        self.mac.cuda()
        if self.agent_mixer != None:
            self.agent_mixer.cuda()
        if self.time_mixer != None:
            self.time_mixer.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.agent_mixer != None:
            th.save(self.agent_mixer.state_dict(), "{}/agent_mixer.th".format(path))
        if self.time_mixer != None:
            th.save(self.time_mixer.state_dict(), "{}/time_mixer.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.agent_mixer != None:
            self.agent_mixer.load_state_dict(th.load("{}/agent_mixer.th".format(path), map_location=lambda storage, loc: storage))
        if self.time_mixer != None:
            self.time_mixer.load_state_dict(th.load("{}/time_mixer.th".format(path), map_location=lambda storage, loc: storage))