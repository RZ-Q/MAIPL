import torch

class ScriptPreferences:
    def __init__(self, args, prefer_mac) -> None:
        self.args = args
        self.prefer_mac = prefer_mac
        self.prefer_mac.load_models(self.args.policy_dir)
        if self.args.use_cuda:
            prefer_mac.cuda()

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
        cum_p = torch.prod(chosen_p * mask * 10 + (1 - mask), dim=1)
        preferences = []
        for i in range(self.args.n_agents):
            for j in range(i + 1, self.args.n_agents):
                # make onehot labels
                labels = 0.5 * (cum_p[:, i] == cum_p[:, j])
                labels += 1.0 * (cum_p[:, i] < cum_p[:, j])
                preferences.append(torch.stack([1-labels, labels],dim=-1))
        return torch.stack(preferences, dim=1)

    def produce_labels(self, batch):
        return self.policy_preference(batch)

