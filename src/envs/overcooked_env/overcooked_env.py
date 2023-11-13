import sys
sys.path.append("/data/user/kouqian/codes/MARLHF/src/")
# TODO: why cannot directly import? just change python version 3.7 -> 3.9
from envs.multiagentenv import MultiAgentEnv
from typing import Any, Union, List
from collections import namedtuple
from easydict import EasyDict
import gym
import copy
import numpy as np
import pygame
import cv2

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    NO_COUNTERS_PARAMS,
)
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# from utils import deep_merge_dicts
from utils.dict2namedtuple import convert

OvercookEnvTimestep = namedtuple(
    "OvercookEnvTimestep", ["obs", "reward", "done", "info"]
)

# n, s = Direction.NORTH, Direction.SOUTH
# e, w = Direction.EAST, Direction.WEST
# stay, interact = Action.STAY, Action.INTERACT
# Action.ALL_ACTIONS: [n, s, e, w, stay, interact]


class OvercookEnv(MultiAgentEnv):
    config = EasyDict(
        dict(
            env_name="cramped_room",
            horizon=400,
            concat_obs=False,
            action_mask=True,
            shape_reward=True,
        )
    )

    def __init__(
        self,
        map_name="cramped_room",
        horizon=400,
        shape_reward=True,
        vision_obs=False,
    ):
        self.map_name = map_name
        self.episode_limit = horizon
        self.horizon = horizon
        # self.concat_obs = concat_obs
        # self.action_mask = action_mask
        self.shape_reward = shape_reward
        self.mdp = OvercookedGridworld.from_layout_name(self.map_name)
        self.base_env = OvercookedEnv.from_mdp(
            self.mdp, horizon=self.horizon, info_level=0
        )
        self.visualizer = StateVisualizer()

        # rightnow overcook environment encoding only support 2 agent game
        self.n_agents = 2
        self.n_actions = len(Action.ALL_ACTIONS)
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

        # set up obs shape
        self.vision_obs = vision_obs
        if self.vision_obs:
            featurize_fn = lambda mdp, state: mdp.lossless_state_encoding(state)
        else:  # use traditional state encode
            mlam = MediumLevelActionManager.from_pickle_or_compute(
                self.mdp, NO_COUNTERS_PARAMS, force_compute=False
            )
            featurize_fn = lambda mdp, state: mdp.featurize_state(state, mlam)
        self.featurize_fn = featurize_fn

        self._episode_steps = 0

    def step(self, actions):
        assert all(
            self.action_space.contains(a) for a in actions
        ), "%r (%s) invalid" % (actions, type(actions))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in actions]

        _, reward, terminated, env_info = self.base_env.step(joint_action)
        reward = np.array([float(reward)])
        self._episode_steps += 1
        self._eval_episode_return += reward

        if self.shape_reward:
            self._eval_episode_return += sum(env_info["shaped_r_by_agent"])
            reward += sum(env_info["shaped_r_by_agent"])

        env_info["policy_agent_idx"] = 0
        env_info["eval_episode_return"] = self._eval_episode_return
        env_info["other_agent_env_idx"] = 1

        return reward, terminated, env_info

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        # TODO: add vision_obs
        # TODO: add obs separation
        agent_obs = self.featurize_fn(self.mdp, self.base_env.state)[agent_id]
        return agent_obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_mdp, dummy_state)[0].shape
        return obs_shape

    def get_state(self):
        """Returns the global state, which is the concat of agents_obs.
        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.concatenate(self.featurize_fn(self.mdp, self.base_env.state))
        return state

    def get_state_size(self):
        """Returns the shape of the state"""
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        state_shape = np.concatenate(self.featurize_fn(dummy_mdp, dummy_state)).shape[0]
        return state_shape

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        # allow all actions in all states for all agents
        actions = self.mdp.get_actions(self.base_env.state)
        avail_actions = []
        for i in range(self.n_actions):
            if Action.INDEX_TO_ACTION[i] in actions[agent_id]:
                avail_actions.append(1)
            else:
                avail_actions.append(0)
        return avail_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        self.base_env.reset()
        self._eval_episode_return = 0
        self.mdp = self.base_env.mdp
        return self.get_obs(), self.get_state()

    def close(self):
        # Note: the real env instance only has a empty close method, only pas
        pass

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def random_action(self):
        return [self.action_space.sample() for _ in range(self.n_agents)]

    def render(self):
        image = self.visualizer.render_state(
            state=self.base_env.state,
            grid=self.base_env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(self.overcooked.state),
        )

        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (528, 464))

        return image

    def get_env_info(self):
        env_info = super().get_env_info()
        return env_info


class OvercookGameEnv:
    config = EasyDict(
        dict(
            env_name="cramped_room",
            horizon=400,
            concat_obs=False,
            action_mask=False,
            shape_reward=True,
        )
    )

    def __init__(self, cfg) -> None:
        # self.args = deep_merge_dicts(self.config, cfg)
        self.env_name = self.args.env_name
        self.horizon = self.args.horizon
        self.concat_obs = self.args.concat_obs
        self.action_mask = self.args.action_mask
        self.shape_reward = self.args.shape_reward
        self.mdp = OvercookedGridworld.from_layout_name(self.env_name)
        self.base_env = OvercookedEnv.from_mdp(
            self.mdp, horizon=self.horizon, info_level=0
        )

        # rightnow overcook environment encoding only support 2 agent game
        self.agent_num = 2
        self.action_dim = len(Action.ALL_ACTIONS)
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        # set up obs shape
        featurize_fn = lambda mdp, state: mdp.lossless_state_encoding(state)
        self.featurize_fn = featurize_fn
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_mdp, dummy_state)[0].shape  # (5, 4, 26)
        obs_shape = (obs_shape[-1], *obs_shape[:-1])  # permute channel first
        if self.concat_obs:
            obs_shape = (obs_shape[0] * 2, *obs_shape[1:])
        else:
            obs_shape = (2,) + obs_shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=obs_shape, dtype=np.int64
        )
        if self.action_mask:
            self.observation_space = gym.spaces.Dict(
                {
                    "agent_state": self.observation_space,
                    "action_mask": gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.agent_num, self.action_dim),
                        dtype=np.int64,
                    ),
                }
            )

        self.reward_space = gym.spaces.Box(
            low=0, high=100, shape=(1,), dtype=np.float32
        )

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        # Note: the real env instance only has a empty close method, only pass
        pass

    def random_action(self):
        return [self.action_space.sample() for _ in range(self.agent_num)]

    def step(self, action):
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, env_info = self.base_env.step(joint_action)

        reward = np.array([float(reward)])
        self._eval_episode_return += reward
        if self.shape_reward:
            self._eval_episode_return += sum(env_info["shaped_r_by_agent"])
            reward += sum(env_info["shaped_r_by_agent"])
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, next_state)
        ob_p0, ob_p1 = self.obs_preprocess(ob_p0), self.obs_preprocess(ob_p1)
        if self.agent_idx == 0:
            both_agents_ob = [ob_p0, ob_p1]
        else:
            both_agents_ob = [ob_p1, ob_p0]
        if self.concat_obs:
            both_agents_ob = np.concatenate(both_agents_ob)
        else:
            both_agents_ob = np.stack(both_agents_ob)

        env_info["policy_agent_idx"] = self.agent_idx
        env_info["eval_episode_return"] = self._eval_episode_return
        env_info["other_agent_env_idx"] = 1 - self.agent_idx

        action_mask = self.getaction_mask()
        if self.action_mask:
            obs = {"agent_state": both_agents_ob, "action_mask": action_mask}
        else:
            obs = both_agents_ob
        return OvercookEnvTimestep(obs, reward, done, env_info)

    def obs_preprocess(self, obs):
        obs = obs.transpose(2, 0, 1)
        return obs

    def reset(self):
        self.base_env.reset()
        self._eval_episode_return = 0
        self.mdp = self.base_env.mdp
        # random init agent index
        self.agent_idx = np.random.choice([0, 1])
        # fix init agent index
        self.agent_idx = 0
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, self.base_env.state)
        ob_p0, ob_p1 = self.obs_preprocess(ob_p0), self.obs_preprocess(ob_p1)

        if self.agent_idx == 0:
            both_agents_ob = [ob_p0, ob_p1]
        else:
            both_agents_ob = [ob_p1, ob_p0]
        if self.concat_obs:
            both_agents_ob = np.concatenate(both_agents_ob)
        else:
            both_agents_ob = np.stack(both_agents_ob)

        action_mask = self.getaction_mask()

        if self.action_mask:
            obs = {"agent_state": both_agents_ob, "action_mask": action_mask}
        else:
            obs = both_agents_ob
        return obs

    def get_available_actions(self):
        return self.mdp.get_actions(self.base_env.state)

    def getaction_mask(self):
        available_actions = self.get_available_actions()

        action_masks = np.zeros((self.agent_num, self.action_dim)).astype(np.int64)

        for i in range(self.action_dim):
            if Action.INDEX_TO_ACTION[i] in available_actions[0]:
                action_masks[0][i] = 1
            if Action.INDEX_TO_ACTION[i] in available_actions[1]:
                action_masks[1][i] = 1

        return action_masks

    def __repr__(self):
        return "DI-engine Overcooked GameEnv"
