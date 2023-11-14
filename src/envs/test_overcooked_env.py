from time import time
from easydict import EasyDict
import pytest
import numpy as np
from overcooked_env import OvercookEnv


@pytest.mark.envtest
class TestOvercooked:

    @pytest.mark.parametrize("action_mask", [True, False])
    def test_overcook(self, action_mask):
        num_agent = 2
        sum_rew = 0.0
        env = OvercookEnv(EasyDict({'concat_obs': True, 'action_mask': action_mask}))
        obs = env.reset()
        for _ in range(env._horizon):
            action = env.random_action()
            timestep = env.step(action)
            obs = timestep.obs
            if action_mask:
                for k, v in obs.items():
                    if k not in ['agent_state', 'action_mask']:
                        assert False
                    assert v.shape == env.observation_space[k].shape
            else:
                assert obs.shape == env.observation_space.shape
        assert timestep.done
        sum_rew += timestep.info['eval_episode_return'][0]
        print("sum reward is:", sum_rew)

    # @pytest.mark.parametrize("concat_obs", [True, False])
    # def test_overcook_game(self, concat_obs):
    #     env = OvercookGameEnv(EasyDict({'concat_obs': concat_obs}))
    #     print('observation space: {}'.format(env.observation_space.shape))
    #     obs = env.reset()
    #     for _ in range(env._horizon):
    #         action = env.random_action()
    #         timestep = env.step(action)
    #         obs = timestep.obs
    #         assert obs.shape == env.observation_space.shape
    #     assert timestep.done
    #     # TODO: timestep.info is a dict, perhaps sparse_r_by_agent/shaped_r_by_agent
    #     print("agent 0 sum reward is:", timestep.info['eval_episode_return'])
    #     print("agent 1 sum reward is:", timestep.info['eval_episode_return'])
    #     # print("agent 0 sum reward is:", timestep.info[0]['eval_episode_return'])
    #     # print("agent 1 sum reward is:", timestep.info[1]['eval_episode_return'])

def main():
    env = OvercookEnv()
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            # env.render()  # Uncomment for rendering

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            # print(actions, reward)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()

main()