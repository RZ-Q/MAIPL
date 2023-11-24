import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.model import fanin_init

class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)

        # init weight
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # just pass hidden_state
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, h_in