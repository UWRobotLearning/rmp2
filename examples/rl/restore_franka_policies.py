import argparse

import ray
from ray.tune.tune import run_experiments
from rmp2.utils.rllib_utils import register_envs_and_models

import natsort
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--stop-iters", type=int, default=500)
parser.add_argument("--checkpoint-freq", type=int, default=1)

parser.add_argument("--ckpt-path", type=str)
parser.add_argument("--n-workers", type=int, default=None)

if __name__ == "__main__":
    args = parser.parse_args()

    ckpt_dirs = [name for name in os.listdir(args.ckpt_path) if os.path.isdir(os.path.join(args.ckpt_path, name))]
    ckpt_dirs = natsort.natsorted(ckpt_dirs,reverse=True)
    ckpt_number = int(ckpt_dirs[0].split('_')[1])
    ckpt_dir =os.path.join(
        args.ckpt_path,
        ckpt_dirs[0])

    agent_ckpt_path = os.path.join(
        ckpt_dir, 
        'checkpoint-{}'.format(ckpt_number))

    print('checkpoint number', ckpt_number)

    params_file = os.path.join(args.ckpt_path, "params.json")
    with open(params_file) as f:
        config = json.load(f)

    env = config['env']
    if 'hidden_units' in config['model']['custom_model_config']['policy_config']:
        nn_size = config['model']['custom_model_config']['policy_config']['hidden_units'][0]
    else:
        nn_size = config['model']['custom_model_config']['policy_config']['units'][0]

    lr = config['lr']
    clip_param = config['clip_param']
    lambd = config['lambda']
    batch_size = config['train_batch_size']
    obs_num = config['env_config']['max_obstacle_num']

    
    experiment_name = env + '-nn-' + str(nn_size) + '-lr-' + str(lr) + \
        '-cp-' + str(clip_param) + '-l-' + str(lambd) + '-bs-' + str(batch_size) +\
        '-obs-num-' + str(obs_num)

    if args.n_workers is not None:
        config['num_workers'] = args.n_workers


    ray.init()

    register_envs_and_models()


    stop = {
        "training_iteration": args.stop_iters,
    }

    experiments = {
        experiment_name: {
            "run": args.run,
            "checkpoint_freq": args.checkpoint_freq,
            "checkpoint_at_end": True,
            "stop": stop,
            "config": config,
            "restore": agent_ckpt_path,
        }
    }

    run_experiments(experiments, reuse_actors=True, concurrent=True)

    ray.shutdown()