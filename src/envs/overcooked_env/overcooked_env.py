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
    def __init__(
        self,
        map_name="cramped_room",
        horizon=400,
        shape_reward=True,
        vision_obs=False,
        seed=None
    ):
        self.map_name = map_name
        self.episode_limit = horizon
        self.horizon = horizon
        self._seed = seed
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
        # TODO: actions valid check
        # assert all(
        #     self.action_space.contains(a) for a in actions
        # ), "%r (%s) invalid" % (actions, type(actions))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in actions]
        new_env_info = {}

        _, reward, terminated, env_info = self.base_env.step(joint_action)
        reward = np.array([float(reward)])
        self._episode_steps += 1
        self._eval_episode_return += reward

        if self.shape_reward:
            self._eval_episode_return += sum(env_info["shaped_r_by_agent"])
            reward += sum(env_info["shaped_r_by_agent"])

        # env_info["policy_agent_idx"] = 0
        env_info["eval_episode_return"] = self._eval_episode_return
        # env_info["other_agent_env_idx"] = 1
        # new_env_info["shaped_r_by_agent"] = env_info["shaped_r_by_agent"]
        new_env_info["shaped_r1"] = env_info["shaped_r_by_agent"][0] + env_info["sparse_r_by_agent"][0]
        new_env_info["shaped_r2"] = env_info["shaped_r_by_agent"][1] + env_info["sparse_r_by_agent"][1]
        new_env_info["eval_episode_return"] = self._eval_episode_return

        return reward, terminated, new_env_info

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
        obs_shape = self.featurize_fn(dummy_mdp, dummy_state)[0].shape[0]
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
        self._episode_steps = 0
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
    
    def get_indi_terminated(self):
        """Returns the terminated of all agents in a list."""
        terminate = []
        if self._episode_steps < self.horizon:
            return [0, 0]
        else:
            return [1, 1]
