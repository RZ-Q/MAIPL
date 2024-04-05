import os
import torch
import numpy as np

from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from envs.env_wrappers import ShareDummyVecEnv
from utils.logger import setup_logger_kwargs, Logger
from utils.util import evaluate
from datasets.offline_dataset import ReplayBuffer
from datasets.preference_dataset import PrefDataset
# from algos.OMIGA import OMIGA
from algos.MAIPL_OMIGA import MAIPL_OMIGA
from algos.PbRL_OMIGA import PbRL_OMIGA
from algos.MACPL_OMIGA import MACPL_OMIGA

import wandb
from tqdm import tqdm

def make_train_env(config):
    def get_env_fn(rank):
        def init_env():
            if config['env_name'] == "mujoco":
                env_args = {"scenario": config['scenario'],
                            "agent_conf": config['agent_conf'],
                            "agent_obsk": config['agent_obsk'],
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + config['env_name'] + "environment.")
                raise NotImplementedError
            env.seed(config['seed'])
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])


def make_eval_env(config):
    def get_env_fn(rank):
        def init_env():
            if config['env_name'] == "mujoco":
                env_args = {"scenario": config['scenario'],
                            "agent_conf": config['agent_conf'],
                            "agent_obsk": config['agent_obsk'],
                            "episode_limit": 1000}
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + config['env_name'] + "environment.")
                raise NotImplementedError
            env.seed(config['seed'])
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])


def run(config):
    # assert config['algo']=='OMIGA', "Invalid algorithm"
    assert config['env_name'] == 'mujoco', "Invalid environment"
    env_name = config['scenario'] + '-' + config['agent_conf'] + '-' + config['data_type']
    exp_name = "OMIGA"
    name = 'test_s' + str(config['seed'])
    group_name = env_name + '-' + config['algo'] + '-' + str(config["batch_size"]) + '-' + str(config["pref_batch_size"]) + '-' + str(config["alpha"]) + \
        '-' + "usevt" + str(config["v_target"]) + '-' + 'onlyp' + str(config["use_pref_only"])

    if config['wandb'] == True:
        wandb.init(project=exp_name, name=name, group=group_name)

    # Seeding
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 

    env = make_train_env(config)
    eval_env = make_eval_env(config)

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    n_agents = len(env.observation_space)
    print('state_dim:', state_dim, 'action_dim:', action_dim, 'num_agents:', n_agents)

    segment_length = config['segment_length']
    num_feedbacks = config['num_feedbacks']
    pref_dir = config['data_dir'] + env_name + "-" + str(num_feedbacks) + "-" + str(segment_length) + "pref_index.npy"

    logger_kwargs = setup_logger_kwargs(env_name, config['seed'], datestamp=True)
    logger = Logger(**logger_kwargs)
    if config['log_data'] == True:
        logger.save_config(config)
    
    # Datasets
    offline_dataset = ReplayBuffer(state_dim, action_dim, n_agents, env_name, config['data_dir'], device=config['device'])
    pref_dataset = PrefDataset(state_dim, action_dim, n_agents, env_name, config['data_dir'], pref_dir, segment_length, config, device=config['device'])
    # offline_dataset.load() process datas to single-step (s,a,s')
    offline_dataset.load()
    pref_dataset.load()

    result_logs = {}

    def _eval_and_log(train_result, config):
        train_result = {k: v.detach().cpu().numpy() for k, v in train_result.items()}
        print('\n==========Policy testing==========')
        # evaluation via real-env rollout
        ep_r = evaluate(agent, eval_env, config['env_name'])

        train_result.update({'ep_r': ep_r})

        result_log = {'log': train_result, 'step': iteration}
        result_logs[str(iteration)] = result_log

        for k, v in sorted(train_result.items()):
            print(f'- {k:23s}:{v:15.10f}')
        print(f'iteration={iteration}')
        print('\n==========Policy training==========', flush=True)

        return train_result
     
    # Agent
    # agent = MAIPL_OMIGA(state_dim, action_dim, n_agents, eval_env, config)
    # agent = PbRL_OMIGA(state_dim, action_dim, n_agents, eval_env, config)
    agent = MACPL_OMIGA(state_dim, action_dim, n_agents, eval_env, config)

    # Train
    print('\n==========Start training==========')

    for iteration in tqdm(range(0, config['total_iterations']), ncols=70, desc=config['algo'], initial=1, total=config['total_iterations'], ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
        offline_batch = offline_dataset.sample(config['batch_size'])
        pref_batch = pref_dataset.sample(config['pref_batch_size'])
        train_result = agent.train_step(offline_batch, pref_batch, iteration)
        if iteration % config['log_iterations'] == 0:
            train_result = _eval_and_log(train_result, config)
            if config['wandb'] == True:
                wandb.log(train_result)

    # Save results
    if config['log_data'] == True:
        logger.save_result_logs(result_logs)

    env.close()
    eval_env.close()

if __name__ == "__main__":
    from configs.config import get_parser
    args = get_parser().parse_args() 
    run(vars(args))