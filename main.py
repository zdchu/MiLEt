import argparse
import warnings

import numpy as np
import torch
import IPython
import random
from copy import copy

from config.mujoco import args_ant_goal_milet, args_walker_milet, args_humanoid_dir_milet, args_hopper_milet
from environments.parallel_envs import make_vec_envs
from exp_metalearner import Explore_MetaLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='ant_goal_milet')
    args, rest_args = parser.parse_known_args()
    env = args.env_type
    
    # Ant goal
    if env == 'ant_goal_milet':
        args = args_ant_goal_milet.get_args(rest_args)

    # Walker
    elif env == 'walker_milet':
        args = args_walker_milet.get_args(rest_args)
    
    # Human
    elif env == 'humanoid_dir_milet':
        args = args_humanoid_dir_milet.get_args(rest_args)

    # Hopper
    elif env == 'hopper_milet':
        args = args_hopper_milet.get_args(rest_args)

    if not hasattr(args, 'k'):
        args.k = 4   

    if not hasattr(args, 'env_names'):
        args.env_names = None
        args.test_env_names = None

    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None, env_names=args.env_names, sparse=args.sparse_reward
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]
    

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False


    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True

    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None
        learner = Explore_MetaLearner(args)
        learner.train()

if __name__ == '__main__':
    main()
