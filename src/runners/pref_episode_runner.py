from envs import REGISTRY as env_REGISTRY
from functools import partial
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
import torch.nn.functional as F

class PrefEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_returns_hat = []
        self.test_returns_hat = []
        self.train_stats = {}
        self.test_stats = {}
        # individual returns log
        self.train_indi_returns = []
        self.test_indi_returns = []

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, random=False, reward_model=None):
        self.reset()

        terminated = False
        episode_return = 0
        episode_return_hat = 0
        episode_indi_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # Random actions
            if random:
                avail_actions = self.batch["avail_actions"][:, self.t]
                actions = Categorical(avail_actions.float()).sample().long()
            else:
                # epsilon is not need in sac
                # in qmix, enlarge decay steps is ok 50000->60000
                actions = self.mac.select_actions(
                    self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
                )

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            episode_indi_return += np.array(env_info["indi_reward"])
            

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "indi_terminated": [self.env.get_indi_terminated()],
                "indi_reward": list(env_info["indi_reward"]),
            }

            # delete indi_reward in env_info, thus it will not in cur_stats
            del env_info["indi_reward"]

            self.batch.update(post_transition_data, ts=self.t)
            if reward_model is not None:
                if self.args.actions_onehot:
                    a = F.one_hot(actions, self.args.n_actions)
                else:
                    a = actions.unsqueeze(-1)
                
                obs = torch.tensor(np.array(pre_transition_data['obs'])).to(self.args.device)
                ids = torch.eye(self.args.n_agents).unsqueeze(0).to(self.args.device)
                if self.args.state_or_obs:
                    s = torch.tensor(np.array(pre_transition_data['state'])).to(self.args.device)
                    global_sa = torch.cat((s, a.view(1, -1)), dim=-1)
                else:
                    global_sa = torch.cat((obs, ids, a), dim=-1).view(1, -1)   
                sa = torch.cat((obs, ids, a), dim=-1)
                reward_hat = reward_model.r_hat(global_sa)
                if self.args.use_local_reward:
                    indi_reward_hat = reward_model.local_r_hat(sa)
                    self.batch.update({"indi_reward_hat": indi_reward_hat[0].squeeze(-1)}, ts=self.t)
                else:
                    self.batch.update({"indi_reward_hat": [0 for _ in range(self.args.n_agents)]}, ts=self.t)
                episode_return_hat += reward_hat[0][0].item()
                self.batch.update({"reward_hat": [(reward_hat[0][0].item(),)]}, ts=self.t)
            else:
                episode_return_hat += 0
                self.batch.update({"reward_hat": [(0,)]}, ts=self.t)
                self.batch.update({"indi_reward_hat": [0 for _ in range(self.args.n_agents)]}, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_returns_hat = self.test_returns_hat if test_mode else self.train_returns_hat
        cur_indi_returns = self.test_indi_returns if test_mode else self.train_indi_returns
        log_prefix = "test_" if test_mode else ""
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        for k in set(cur_stats) | set(env_info):
            if cur_stats.get(k, 0) == 0:
                cur_stats.update({k: env_info.get(k, 0)})
            else:
                cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        if reward_model is not None:
            cur_stats["total_feedbacks"] = reward_model.get_feedbacks()

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        cur_returns_hat.append(episode_return_hat)
        cur_indi_returns.append(episode_indi_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_returns_hat, cur_indi_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_returns_hat, cur_indi_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, returns_hat, indi_returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        self.logger.log_stat(prefix + "return_hat_mean", np.mean(returns_hat), self.t_env)
        self.logger.log_stat(prefix + "return_hat_std", np.std(returns_hat), self.t_env)
        returns_hat.clear()
        # for i in range(self.args.n_agents):
        #     self.logger.log_stat(prefix + "return_mean" + str(i), np.mean(np.array(indi_returns)[:, i]), self.t_env)
        #     self.logger.log_stat(prefix + "return_std" + str(i), np.std(np.array(indi_returns)[:, i]), self.t_env)
        indi_returns.clear()

        for k, v in stats.items():
            if k != "n_episodes" and k != "total_feedbacks":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
            elif k == "total_feedbacks":
                self.logger.log_stat(prefix + "total_feedbacks", stats["total_feedbacks"], self.t_env)
        stats.clear()
