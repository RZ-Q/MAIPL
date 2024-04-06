import copy
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
import torch as th
from utils.offpg_utils import build_target_q
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop
from modules.mixers.qmix import QMixer
import torch.nn.functional as F
import numpy as np
import wandb

class CPLLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.cpl_lambda = args.cpl_lambda
        self.cpl_alpha = args.cpl_alpha

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.agent_params = list(mac.parameters())

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch0, batch1, t_env, labels):
        # # ----------- batch0 preferred -----------------
        bs = batch0['batch_size']
        actions0 = batch0["actions"][:, :-1]
        terminated0 = batch0["terminated"][:, :-1].float()
        avail_actions0 = batch0["avail_actions"][:, :-1].long()

        mask0 = batch0["filled"][:, :-1].float()
        mask0[:, 1:] = mask0[:, 1:] * (1 - terminated0[:, :-1])
        mask_td0 = mask0.repeat(1, 1, self.n_agents).view(bs, -1, self.n_agents)
        mask0 = mask0.repeat(1, 1, self.n_agents).view(-1)

        # Calculate estimated Q-Values
        mac_out0 = []
        self.mac.init_hidden(batch0['batch_size'])
        for t in range(batch0['max_seq_length'] - 1):
            agent_outs = self.mac.forward(batch0, t=t)
            mac_out0.append(agent_outs)
        mac_out0 = th.stack(mac_out0, dim=1)
        
        mac_out0[avail_actions0 == 0] = 0
        mac_out0 = mac_out0/mac_out0.sum(dim=-1, keepdim=True)
        mac_out0[avail_actions0 == 0] = 0
        #mac_out = F.softmax(mac_out, dim=-1) # get softmax policy

        pi0 = mac_out0.view(-1, self.n_actions)
        pi_taken0 = th.gather(pi0, dim=1, index=actions0.reshape(-1, 1)).squeeze(1)
        pi_taken0[mask0 == 0] = 1.0
        log_pi_taken0 = th.log(pi_taken0)
        log_pi_taken0 = log_pi_taken0.view(bs, -1, self.n_agents).sum(-1)

        # # ----------- batch1 not preferred -----------------
        actions1 = batch1["actions"][:, :-1]
        terminated1 = batch1["terminated"][:, :-1].float()
        avail_actions1 = batch1["avail_actions"][:, :-1].long()

        mask1 = batch1["filled"][:, :-1].float()
        mask1[:, 1:] = mask1[:, 1:] * (1 - terminated1[:, :-1])
        mask_td1 = mask1.repeat(1, 1, self.n_agents).view(bs, -1, self.n_agents)
        mask1 = mask1.repeat(1, 1, self.n_agents).view(-1)

        # Calculate estimated Q-Values
        mac_out1 = []
        self.mac.init_hidden(batch1['batch_size'])
        for t in range(batch1['max_seq_length'] - 1):
            agent_outs = self.mac.forward(batch1, t=t)
            mac_out1.append(agent_outs)
        mac_out1 = th.stack(mac_out1, dim=1)
        
        mac_out1[avail_actions1 == 0] = 0
        mac_out1 = mac_out1/mac_out1.sum(dim=-1, keepdim=True)
        mac_out1[avail_actions1 == 0] = 0
        #mac_out = F.softmax(mac_out, dim=-1) # get softmax policy

        pi1 = mac_out1.view(-1, self.n_actions)
        pi_taken1 = th.gather(pi1, dim=1, index=actions1.reshape(-1, 1)).squeeze(1)
        pi_taken1[mask1 == 0] = 1.0
        log_pi_taken1 = th.log(pi_taken1)
        log_pi_taken1 = log_pi_taken1.view(bs, -1, self.n_agents).sum(-1)

        # # ------------- CPL loss ------------------
        adv0 = (self.cpl_alpha * log_pi_taken0 * mask0).sum(1)
        adv1 = (self.cpl_alpha * log_pi_taken1 * mask1).sum(1)
        logit10 = adv1 - self.cpl_lambda * adv0
        logit01 = adv0 - self.cpl_lambda * adv1
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

    def cuda(self):
        self.mac.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))