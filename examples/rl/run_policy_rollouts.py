"""
Script for visualizing the learned policies
"""

import os
import json
import argparse

import ray
from ray.tune.registry import get_trainable_cls
from rmp2.utils.rllib_utils import register_envs_and_models
from ray.tune.registry import ENV_CREATOR, _global_registry
import numpy as np
from tqdm import tqdm
import natsort


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=str)
parser.add_argument("--ckpt-number", type=int, default=-1)
parser.add_argument("--horizon", type=int, default=1000)
parser.add_argument("--n-trials", type=int, default=1)


if __name__ == "__main__":
    args = parser.parse_args()

    # if checkpoint number is not given, use the latest checkpoint
    if args.ckpt_number < 0:
        print('using the latest checkpoint')
        ckpt_dirs = [name for name in os.listdir(args.ckpt_path) if os.path.isdir(os.path.join(args.ckpt_path, name))]
        ckpt_dirs = natsort.natsorted(ckpt_dirs,reverse=True)
        ckpt_number = int(ckpt_dirs[0].split('_')[1])
        ckpt_dir =os.path.join(
            args.ckpt_path,
            ckpt_dirs[0])
    # otherwise load the specified checkpoint unless it is not found
    else:
        ckpt_dir = os.path.join(
            args.ckpt_path,
            'checkpoint_{}'.format(args.ckpt_number))
        if not os.path.isdir(ckpt_dir):
            Warning('checkpoint number not found, use the latest one')
            ckpt_dirs = [name for name in os.listdir(args.ckpt_path) if os.path.isdir(os.path.join(args.ckpt_path, name))]
            ckpt_dirs = natsort.natsorted(ckpt_dirs,reverse=True)
            ckpt_number = int(ckpt_dirs[0].split('_')[1])
            ckpt_dir =os.path.join(
                args.ckpt_path,
                ckpt_dirs[0])
            print('using the latest checkpoint')
        else:
            ckpt_number = args.ckpt_number
    print('using checkpoint', ckpt_number)

    agent_ckpt_path = os.path.join(
        ckpt_dir, 
        'checkpoint-{}'.format(ckpt_number))

    # load experiment config
    params_file = os.path.join(args.ckpt_path, "params.json")
    with open(params_file) as f:
        config = json.load(f)
    # disable parallelism
    config['num_workers'] = 0

    # initialize ray
    ray.init()

    # register customized environments and models within ray
    register_envs_and_models()

    # get environment creator and enable rendering
    env_creator = _global_registry.get(ENV_CREATOR, config['env'])
    env_config = config['env_config'].copy()
    env_config['render'] = True

    # restored the learned policy
    cls = get_trainable_cls("PPO")
    agent = cls(config, config['env'])
    agent.restore(agent_ckpt_path)

    policy = agent.workers.local_worker().get_policy()
    agent.workers.local_worker().env.__del__()
    model = policy.model

    # create a environment
    env = env_creator(env_config)
    env.seed(100)
    env.reset()
    env.step(np.zeros(env.action_space.shape[0],))

    input('waiting for user input...')

    # roll out the policy for n trials
    for _ in range(args.n_trials):
        state = env.reset()
        rewards = []
        for i in tqdm(range(args.horizon)):
            action = policy.compute_actions([state])
            action = action[0][0]
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        print('total reward from the current episode: ', np.sum(rewards))

        episode_len = i + 1