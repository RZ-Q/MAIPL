import torch

class ScriptPreferences:
    def __init__(self, args, prefer_mac) -> None:
        self.args = args
        self.map_name = args.env_args["map_name"]
        self.preference_type = args.local_preference_type
        if self.preference_type == "policy":
            self.prefer_mac = prefer_mac
            self.prefer_mac.load_models(self.args.policy_dir)
            if self.args.use_cuda:
                prefer_mac.cuda()

    def process_states(self, states: torch.Tensor, actions: torch.Tensor):
        # states shape [bs, ep_len, state_size]
        # actions shape [bs, ep_len, num_agents, 1]
        agent_num = 0
        if self.map_name == "3m":
            agent_num = 3
            agents_health = torch.index_select(
                states, dim=-1, index=torch.tensor([0, 4, 8]).to(self.args.device)
            )
            agents_cooldown = torch.index_select(
                states, dim=-1, index=torch.tensor([1, 5, 9]).to(self.args.device)
            )
            enemies_health = torch.index_select(
                states, dim=-1, index=torch.tensor([12, 15, 18]).to(self.args.device)
            )
            actions = actions.squeeze(-1)  # Remove last dim

        return agent_num, agents_health, agents_cooldown, enemies_health, actions

    def high_health_preference(self, batch):
        """
        This preference prefers high health agents.
        Computation:
            Agent with high Agent_health[-1] value get higher preference
        """
        states = batch["state"][:, :-1] 
        actions = batch["actions"][:, :-1]
        agent_num, agents_health, _, _, _ = self.process_states(states, actions)
        preferences = []
        for i in range(agent_num):
            for j in range(i, agent_num):
                labels = 0.5 * (agents_health[:, -1, i] == agents_health[:, -1, j])
                labels += 1.0 * (agents_health[:, -1, i] < agents_health[:, -1, j]) 
                preferences.append(labels)
        
        return torch.stack(preferences, dim=-1)
    
    def true_indi_rewards_preference(self, batch):
        indi_rewards = batch["indi_reward"][:, :-1]
        agent_num = self.args.n_agents
        preferences = []
        for i in range(agent_num):
            for j in range(i + 1, agent_num):
                # make onehot labels
                labels = 0.5 * (indi_rewards[:, :, i].sum(-1) == indi_rewards[:, :, j].sum(-1))
                labels += 1.0 * (indi_rewards[:, :, i].sum(-1) < indi_rewards[:, :, j].sum(-1))
                preferences.append(torch.stack([1-labels, labels],dim=-1))
        return torch.stack(preferences, dim=1)

    def policy_preference(self, batch):
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        indi_terminated = batch["indi_terminated"][:, :-1].float()

        # Calculate estimated Q-Values
        mac_out = []
        with torch.no_grad(): # reference        
            self.prefer_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs =  self.prefer_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # Concat over time
        mac_out[avail_actions == 0] = -1e10
        mac_out = torch.softmax(mac_out, dim=-1)
        chosen_p = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        cum_p = torch.prod(chosen_p * mask + (1 - mask), dim=1)
        preferences = []
        for i in range(self.args.n_agents):
            for j in range(i + 1, self.args.n_agents):
                # make onehot labels
                labels = 0.5 * (cum_p[:, i] == cum_p[:, j])
                labels += 1.0 * (cum_p[:, i] < cum_p[:, j])
                preferences.append(torch.stack([1-labels, labels],dim=-1))
        return torch.stack(preferences, dim=1)

    def produce_labels(self, batch):
        if self.preference_type == 'high_health':
            return self.high_health_preference(batch)
        elif self.preference_type == "true_indi_rewards":
            return self.true_indi_rewards_preference(batch)
        elif self.preference_type == "policy":
            return self.policy_preference(batch)

