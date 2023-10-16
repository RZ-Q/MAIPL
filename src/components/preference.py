import torch


class ScriptPreferences:
    def __init__(self, args) -> None:
        self.args = args
        self.map_name = args.env_args["map_name"]

    def process_states(self, states: torch.Tensor, actions: torch.Tensor):
        # states shape [bs, ep_len, state_size]
        # actions shape [bs, ep_len, num_agents, 1]
        agent_num = 0
        if self.map_name == "3m":
            agent_num = 3
            agents_health = torch.index_select(
                states, dim=-1, index=torch.tensor([0, 4, 8])
            )
            agents_cooldown = torch.index_select(
                states, dim=-1, index=torch.tensor([1, 5, 9])
            )
            enemies_health = torch.index_select(
                states, dim=-1, index=torch.tensor([12, 15, 18])
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
                # TODO: fix preferences
                if agents_health[:, -1, i] > agents_health[:, -1, j]:
                    preferences.append(0)
                elif agents_health[:, -1, i] < agents_health[:, -1, j]:
                    preferences.append(1)
                else:
                    preferences.append(0.5)
        return preferences

