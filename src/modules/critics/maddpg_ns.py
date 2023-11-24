import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.model import fanin_init

class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

         # init weight
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MADDPGCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCriticNS, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        if self.args.obs_last_action:
            self.input_shape += self.n_actions
        self.output_type = "q"
        self.critics = [MLP(self.input_shape, self.args.hidden_dim, 1) for _ in range(self.n_agents)]

    def forward(self, inputs, critic_idx):
        # remove actions, just pass in concat of [inputs, actions] as inputs
        # update critic one by one, pass in critic idx
        return self.critics[critic_idx](inputs)

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        return input_shape

    def parameters_by_critic(self):
        params = []
        for i in range(self.n_agents):
            params.append(self.critics[i].parameters())
        return params
    
    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, c in enumerate(self.critics):
            c.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()
    