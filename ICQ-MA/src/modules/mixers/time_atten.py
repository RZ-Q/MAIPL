import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeattenMixer(nn.Module):
    def __init__(self, args):
        super(TimeattenMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.traj_feature_rtgs = args.traj_feature_rtgs
        self.state_dim = int(np.prod(args.state_shape))
        self.traj_dim = self.state_dim + self.n_agents + self.traj_feature_rtgs  # state0, action0, rtgs

        # TODO: expand dim
        self.n_query_embedding_layer1 = args.n_query_embedding_layer1
        self.n_query_embedding_layer2 = args.n_query_embedding_layer2
        self.n_key_embedding_layer1 = args.n_key_embedding_layer1
        self.n_head_embedding_layer1 = args.n_head_embedding_layer1
        self.n_head_embedding_layer2 = args.n_head_embedding_layer2
        self.n_attention_head = args.n_attention_head
        self.n_constant_value = args.n_constant_value

        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(nn.Sequential(nn.Linear(self.traj_dim, self.n_query_embedding_layer1),
                                                           nn.ReLU(),
                                                           nn.Linear(self.n_query_embedding_layer1, self.n_query_embedding_layer2)))
        
        self.key_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.key_embedding_layers.append(nn.Linear(self.state_dim, self.n_key_embedding_layer1))


        self.scaled_product_value = np.sqrt(args.n_query_embedding_layer2)

        self.head_embedding_layer = nn.Sequential(nn.Linear(self.traj_dim, self.n_head_embedding_layer1),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_head_embedding_layer1, self.n_head_embedding_layer2))
        
        self.constant_value_layer = nn.Sequential(nn.Linear(self.traj_dim, self.n_constant_value),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_constant_value, 1))


    def forward(self, agent_qs, states, trajs):
        bs, seq_len = agent_qs.size(0), agent_qs.size(1)
        states = states.reshape(-1, self.state_dim)
        trajs = trajs.reshape(-1, self.traj_dim)
        agent_qs = agent_qs.view(bs, 1, -1)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            traj_embedding = self.query_embedding_layers[i](trajs)
            state_embedding = self.key_embedding_layers[i](states)

            # shape: [-1, 1, state_dim]
            traj_embedding = traj_embedding.reshape(-1, 1, self.n_query_embedding_layer2)
            # shape: [-1, state_dim, seq_len]
            state_embedding = state_embedding.reshape(-1, seq_len, self.n_key_embedding_layer1)
            state_embedding = state_embedding.permute(0, 2, 1)

            # shape: [-1, 1, seq_len]
            raw_lambda = th.matmul(traj_embedding, state_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        # shape: [-1, n_attention_head, seq_len]
        q_lambda_list = th.stack(q_lambda_list, dim=1).squeeze(-2)

        # shape: [-1, seq_len, n_attention_head]
        q_lambda_list = q_lambda_list.permute(0, 2, 1)

        # shape: [-1, 1, n_attention_head]
        q_h = th.matmul(agent_qs, q_lambda_list)

        if self.args.qatten_type == 'weighted':
            # shape: [-1, n_attention_head, 1]
            w_h = th.abs(self.head_embedding_layer(trajs))
            w_h = w_h.reshape(-1, self.n_head_embedding_layer2, 1)
            #TODO: add softmax to w_h
            # shape: [-1, 1]
            sum_q_h = th.matmul(q_h, w_h)
            sum_q_h = sum_q_h.reshape(-1, 1)
        else:
            # shape: [-1, 1]
            sum_q_h = q_h.sum(-1)
            sum_q_h = sum_q_h.reshape(-1, 1)

        c = self.constant_value_layer(trajs)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, -1)
        return q_tot
