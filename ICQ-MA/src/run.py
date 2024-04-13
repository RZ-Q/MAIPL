import datetime
import os
import pprint
from textwrap import fill
import time
import math as mth
import numpy as np
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import h5py
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, Best_experience_Buffer
from components.transforms import OneHot
from components.reward_model import RewardModel
import datetime
import wandb


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    unique_token = args.name + '-' + args.env_args['map_name']+ '-' + str(args.pref_segment_pairs) + '-' + args.offline_dataset_quality + '-' + str(args.seed)
    unique_token_wandb = args.name + '-' + args.env_args['map_name']+ '-' + str(args.pref_segment_pairs) + '-' + args.offline_dataset_quality
    if args.name == 'CPL':
        unique_token += '-' + str(args.cpl_lambda) + '-' + str(args.cpl_alpha)
        unique_token_wandb += '-' + str(args.cpl_lambda) + '-' + str(args.cpl_alpha)
    if args.name == 'MACPL':
        unique_token += '-' + str(args.cpl_lambda) + '-' + str(args.cpl_alpha) + '-ns'
        unique_token_wandb += '-' + str(args.cpl_lambda) + '-' + str(args.cpl_alpha) + '-ns'
    if args.use_reward_hat:
        unique_token += '-' + args.model_type
        unique_token_wandb += '-' + args.model_type
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    if args.use_wandb:
        wandb.init(project="ICQ", group=unique_token_wandb, name=str(args.seed))

    logger.setup_sacred(_run)

    run_sequential(args=args, logger=logger)

    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        episode_batch = runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_shape = env_info["obs_shape"]
    args.state_shape = env_info["state_shape"]
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = -100000
    model_save_time = 0
    start_time = time.time()
    last_time = start_time
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))    
    episode_num = 0


    # # -------------------------- load pref dataset -----------------
    pref_dataset = th.load(args.offline_dataset_dir)
    if args.use_reward_hat:
        pref_dataset['reward'][:, :args.max_traj_length] = th.load(args.offline_dataset_dir[:-3] + '_' + args.model_type + '.th')

    if args.train_reward:
        # ----------------------------train reward-------------------------------
        reward_model = RewardModel(args)
        preds = reward_model.train(pref_dataset)
        #TODO: add reward norm
        reward_save_path = args.offline_dataset_dir[:-3] + '_' + args.model_type + '.th'
        th.save(preds, reward_save_path)
    else:
        # ----------------------------train-------------------------------
        while runner.t_env <= args.t_max:
            if runner.t_env >= 4000200:
                break

            th.set_num_threads(8)

            running_log = {}

            if not args.use_pref:
                # # --------------------------- sample for ICQ,BC,OMIGA -------------------------------
                sample_number = np.random.choice(len(pref_dataset['action']), args.off_batch_size, replace=False)
                filled_sample = pref_dataset['filled'][sample_number]
                max_ep_t_h = filled_sample.sum(1).max(0)[0]
                filled_sample = filled_sample[:, :max_ep_t_h]
                actions_sample = pref_dataset['action'][sample_number][:, :max_ep_t_h]
                actions_onehot_sample = th.nn.functional.one_hot(actions_sample, num_classes=args.n_actions).squeeze(-2)
                avail_actions_sample = pref_dataset['avail_action'][sample_number][:, :max_ep_t_h]
                obs_sample = pref_dataset['obs'][sample_number][:, :max_ep_t_h]
                reward_sample = pref_dataset['reward'][sample_number][:, :max_ep_t_h]
                state_sample = pref_dataset['state'][sample_number][:, :max_ep_t_h]
                terminated_sample = pref_dataset['terminated'][sample_number][:, :max_ep_t_h]
                mask_sample = pref_dataset['mask'][sample_number][:, :max_ep_t_h]


                off_batch = {}
                off_batch['obs'] = obs_sample.to(args.device)
                off_batch['reward'] = reward_sample.to(args.device)
                off_batch['actions'] = actions_sample.to(args.device)
                off_batch['actions_onehot'] = actions_onehot_sample.to(args.device)
                off_batch['avail_actions'] = avail_actions_sample.to(args.device)
                off_batch['filled'] = filled_sample.to(args.device)
                off_batch['state'] = state_sample.to(args.device)
                off_batch['terminated'] = terminated_sample.to(args.device)
                off_batch['max_seq_length'] = max_ep_t_h.to(args.device)
                off_batch['mask'] = mask_sample.to(args.device)
                off_batch['batch_size'] = args.off_batch_size
            
            else:
                # # --------------------------- sample for pref method -------------------------------
                sample_number0 = np.random.choice(int(len(pref_dataset['action']) / 2), int(args.off_batch_size / 2), replace=False)
                sample_number1 = sample_number0 + int(len(pref_dataset['action']) / 2)

                filled_sample = pref_dataset['filled'][sample_number0]
                max_ep_t_h = filled_sample.sum(1).max(0)[0]
                filled_sample = filled_sample[:, :max_ep_t_h]
                actions_sample = pref_dataset['action'][sample_number0][:, :max_ep_t_h]
                actions_onehot_sample = th.nn.functional.one_hot(actions_sample, num_classes=args.n_actions).squeeze(-2)
                avail_actions_sample = pref_dataset['avail_action'][sample_number0][:, :max_ep_t_h]
                obs_sample = pref_dataset['obs'][sample_number0][:, :max_ep_t_h]
                reward_sample = pref_dataset['reward'][sample_number0][:, :max_ep_t_h]
                state_sample = pref_dataset['state'][sample_number0][:, :max_ep_t_h]
                terminated_sample = pref_dataset['terminated'][sample_number0][:, :max_ep_t_h]

                off_batch0 = {}
                off_batch0['obs'] = obs_sample.to(args.device)
                off_batch0['reward'] = reward_sample.to(args.device)
                off_batch0['actions'] = actions_sample.to(args.device)
                off_batch0['actions_onehot'] = actions_onehot_sample.to(args.device)
                off_batch0['avail_actions'] = avail_actions_sample.to(args.device)
                off_batch0['filled'] = filled_sample.to(args.device)
                off_batch0['state'] = state_sample.to(args.device)
                off_batch0['terminated'] = terminated_sample.to(args.device)
                off_batch0['max_seq_length'] = max_ep_t_h.to(args.device)
                off_batch0['batch_size'] = int(args.off_batch_size / 2)

                filled_sample = pref_dataset['filled'][sample_number1]
                max_ep_t_h = filled_sample.sum(1).max(0)[0]
                filled_sample = filled_sample[:, :max_ep_t_h]
                actions_sample = pref_dataset['action'][sample_number1][:, :max_ep_t_h]
                actions_onehot_sample = th.nn.functional.one_hot(actions_sample, num_classes=args.n_actions).squeeze(-2)
                avail_actions_sample = pref_dataset['avail_action'][sample_number1][:, :max_ep_t_h]
                obs_sample = pref_dataset['obs'][sample_number1][:, :max_ep_t_h]
                reward_sample = pref_dataset['reward'][sample_number1][:, :max_ep_t_h]
                state_sample = pref_dataset['state'][sample_number1][:, :max_ep_t_h, 0]
                terminated_sample = pref_dataset['terminated'][sample_number1][:, :max_ep_t_h]

                off_batch1 = {}
                off_batch1['obs'] = obs_sample.to(args.device)
                off_batch1['reward'] = reward_sample.to(args.device)
                off_batch1['actions'] = actions_sample.to(args.device)
                off_batch1['actions_onehot'] = actions_onehot_sample.to(args.device)
                off_batch1['avail_actions'] = avail_actions_sample.to(args.device)
                off_batch1['filled'] = filled_sample.to(args.device)
                off_batch1['state'] = state_sample.to(args.device)
                off_batch1['terminated'] = terminated_sample.to(args.device)
                off_batch1['max_seq_length'] = max_ep_t_h.to(args.device)
                off_batch1['batch_size'] = int(args.off_batch_size / 2)

            # --------------------- ICQ-MA --------------------------------
            if args.name == "ICQ-MA":
                learner.train_critic(off_batch, best_batch=None, running_log=running_log, t_env=runner.t_env)
                learner.train(off_batch, runner.t_env, running_log)
            # --------------------- BC --------------------------------
            elif args.name == "BC":
                learner.train(off_batch, runner.t_env, running_log)
            # --------------------- CPL --------------------------------
            elif args.name == "CPL" or args.name == "MACPL":
                learner.train(off_batch0, off_batch1, runner.t_env, pref_dataset['labels'][sample_number0].to(args.device), running_log)
            
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0: # args.test_interval

                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()

                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    runner.run(test_mode=True, running_log=running_log)

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

            episode += args.batch_size_run

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                running_log.update({"episode": episode})
                if args.use_wandb:
                    wandb.log(running_log)
                logger.print_recent_stats()
                last_log_T = runner.t_env
            
            episode_num += 1
            runner.t_env += 100

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def process_batch(batch, args):

    if batch.device != args.device:
        batch.to(args.device)
    return batch