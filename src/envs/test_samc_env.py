from starcraft2_mappo import StarCraft2Env
import numpy as np
import parser
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='smac_mappo')
    parser.add_argument('--map_name', type=str, default='MMM2',
                        help="Which smac map to run on")
    parser.add_argument('--units', type=str, default='10v10') # for smac v2
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    env = StarCraft2Env(args=args, seed=42)
    local_obs, global_state, available_actions = env.reset()
    env_info = env.get_env_info()

main()