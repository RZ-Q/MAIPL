import torch


class ScriptPreferences:
    def __init__(self, args) -> None:
        self.args = args
        self.map_name = args.env_args["map_name"]
        self.preference_type = args.preference_type

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

    def high_health_preference(self, states: torch.Tensor, actions: torch.Tensor):
        """
        This preference prefers high health agents.
        Computation:
            Agent with high Agent_health[-1] value get higher preference
        """
        agent_num, agents_health, _, _, _ = self.process_states(states, actions)
        preferences = []
        for i in range(agent_num):
            for j in range(i, agent_num):
                labels = 0.5 * (agents_health[:, -1, i] == agents_health[:, -1, j])
                labels += 1.0 * (agents_health[:, -1, i] < agents_health[:, -1, j]) 
                preferences.append(labels)
        
        return torch.stack(preferences, dim=-1)
    
    def true_indi_rewards_preference(self, indi_rewards):
        agent_num = self.args.n_agents
        preferences = []
        for i in range(agent_num):
            for j in range(i + 1, agent_num):
                labels = 0.5 * (indi_rewards[:, :, i].sum(-1) == indi_rewards[:, :, j].sum(-1))
                labels += 1.0 * (indi_rewards[:, :, i].sum(-1) < indi_rewards[:, :, j].sum(-1)) 
                preferences.append(labels)
        return preferences

    def produce_labels(self, states: torch.Tensor, actions: torch.Tensor, indi_rewards: torch.Tensor):
        if self.preference_type == 'high_health':
            return self.high_health_preference(states, actions)
        elif self.preference_type == "true_indi_rewards":
            return self.true_indi_rewards_preference(indi_rewards)

