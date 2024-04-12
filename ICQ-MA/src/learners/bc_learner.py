from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
# from components.standarize_stream import RunningMeanStd
import wandb


class BCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        
        self.log_stats_t = -self.args.learner_log_interval - 1


    def train(self, batch, t_env, running_log):
        # Get the relevant quantities
        bs = batch['batch_size']
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1].long()

        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_td = mask.repeat(1, 1, self.n_agents).unsqueeze(-1)
        # mask = mask.repeat(1, 1, self.n_agents).view(-1)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch['batch_size'])
        for t in range(batch['max_seq_length'] - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0
        #mac_out = F.softmax(mac_out, dim=-1) # get softmax policy

        pi_taken = th.gather(mac_out, dim=-1, index=actions)
        pi_taken[mask_td == 0] = 1.0
        log_pi_taken = th.log(pi_taken)
        loss = - (log_pi_taken * mask_td).sum() / mask_td.sum()


        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env
            running_log.update({
                "loss": loss.item(),
                "agent_grad_norm": grad_norm,
            })


    def cuda(self):
        self.mac.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))