from starcraft2v2 import StarCraft2Env
import numpy as np


def main():
    env = StarCraft2Env(map_name="3m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        env._kill_all_units()
        env.init_units(ally_team=["marine"], enemy_team=["marine"])
        action = [6,1,1]
        reward, terminated, _ = env.step(action)

        # while not terminated:
        #     obs = env.get_obs()
        #     state = env.get_state()
        #     # env.render()  # Uncomment for rendering

        #     actions = []
        #     for agent_id in range(n_agents):
        #         avail_actions = env.get_avail_agent_actions(agent_id)
        #         avail_actions_ind = np.nonzero(avail_actions)[0]
        #         action = np.random.choice(avail_actions_ind)
        #         actions.append(action)

        #     reward, terminated, _ = env.step(actions)
        #     episode_reward += reward

        # print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()

main()