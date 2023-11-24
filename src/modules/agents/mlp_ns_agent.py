from typing import Iterator
import torch.nn as nn
from torch.nn.parameter import Parameter
from modules.agents.mlp_agent import MLPAgent
import torch as th

class MLPNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPNSAgent, self).__init__()
        self.args = args
        self.n_agents = self.args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([MLPAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
    
    def get_param_by_agent(self):
        params = []
        for a in self.agents:
            params.append(a.parameters())
        return params
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return super().parameters(recurse)