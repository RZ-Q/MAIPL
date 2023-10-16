# record base_agent func
'''
 def generate_prompt(self, goal, subgoals, deque_obs, deque_actions):
        ldo = len(deque_obs)
        lda = len(deque_actions)

        head_prompt = "Possible action of the agent:"
        for sg in subgoals:
            head_prompt += " {},".format(sg)
        head_prompt = head_prompt[:-1]

        g = " \n Goal of the agent: {}".format(goal)
        obs = ""
        for i in range(ldo):
            obs += " \n Observation {}: ".format(i)
            for d_obs in deque_obs[i]:
                obs += "{}, ".format(d_obs)
            obs += "\n Action {}: ".format(i)
            if i < lda:
                obs += "{}".format(deque_actions[i])
        return head_prompt + g + obs
'''
import gym
import warnings
warnings.filterwarnings('ignore')

import babyai_text
env = gym.make("BabyAI-MixedTrainLocal-v0")

n_actions = env.action_space

state = env.reset()
print(state)
