import argparse

import ray
from ray.tune import grid_search
from ray.tune.tune import run_experiments
from rmp2.utils.rllib_utils import register_envs_and_models
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--experiment-name", type=str, default=None)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=500)
parser.add_argument("--checkpoint-freq", type=int, default=1)

parser.add_argument("--env", type=str, default='3link_residual')

parser.add_argument("--nn-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--clip-param", type=float, default=0.2)
parser.add_argument("--lambd", type=float, default=0.99)
parser.add_argument("--batch-size", type=int, default=67312) # 336560
parser.add_argument("--sgd-minibatch-size", type=int, default=4096) # 336560
parser.add_argument("--n-seeds", type=int, default=4)

parser.add_argument("--fixed-goal", action='store_true')
parser.add_argument("--fixed-init", action='store_true')
parser.add_argument("--fixed-obs", action='store_true')
parser.add_argument("--obs-free", action='store_true')

parser.add_argument("--goal-reward-model", type=str, default="gaussian")
parser.add_argument("--goal-angle-range", type=float, default=2)
parser.add_argument("--goal-minor-radius", type=float, default=0.375)
parser.add_argument("--obs-num", type=int, default=3)

parser.add_argument("--n-workers", type=int, default=10)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.experiment_name is None:
        experiment_name = args.env + '-nn-' + str(args.nn_size) + '-lr-' + str(args.lr) + \
            '-cp-' + str(args.clip_param) + '-l-' + str(args.lambd) + '-bs-' + str(args.batch_size) +\
            '-obs-num' + str(args.obs_num) + '-ar-' + str(args.goal_angle_range) + \
            '-mr-' + str(args.goal_minor_radius)
        if args.fixed_goal:
            experiment_name += '-fixed_goal'
        if args.fixed_init:
            experiment_name += '-fixed_init'
        if args.fixed_obs and not args.obs_free:
            experiment_name += '-fixed_obs'
        if args.obs_free:
            experiment_name += '-obs_free'
        if not args.goal_reward_model == "gaussian":
            experiment_name += '-' + args.goal_reward_model
    else:
        experiment_name = args.experiment_name


    ray.init()

    register_envs_and_models()

    env_config = {
        "horizon": 1800,
        "max_obstacle_num": args.obs_num,
        "min_obstacle_num": args.obs_num,
        "goal_reward_model": args.goal_reward_model,
        "goal_torus_angle_center": np.pi, 
        "goal_torus_angle_range": args.goal_angle_range * np.pi, 
        "goal_torus_minor_radius": args.goal_minor_radius,
    }
    if args.fixed_goal:
        env_config['goal'] = [0.25, 0.5]
    if args.fixed_init:
        env_config['q_init'] = [0., 0., 0.]
    if args.fixed_obs and not args.obs_free:
        env_config['obstacle_configs'] = [[{'center': [0.0, 0.5], 'radius': 0.1}]]
    if args.obs_free:
        env_config['obstacle_configs'] = [[]]

    config = {
        "env": args.env,
        "env_config": env_config,
        "model": {
            "custom_model": "diag_gaussian_model",
            "custom_model_config": {
                "policy_config": {
                    "model": "mlp",
                    "hidden_units": (args.nn_size, args.nn_size // 2),
                    "activation": "relu",
                    "hidden_layer_init_scale": 2.0,
                    "output_layer_init_scale": 0.1,
                },
                "value_config": {
                    "model": "mlp",
                    "hidden_units": (256, 128)
                },
                "env_wrapper": args.env,
                "init_lstd": -1.,
                "min_std": 1e-4,
                "dtype": "float32",
            },
        },
        "batch_mode": "complete_episodes",
        "rollout_fragment_length": 600,
        "train_batch_size": args.batch_size,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "lr": args.lr,
        "lambda": args.lambd,
        "clip_param": args.clip_param,
        "vf_clip_param": 500.0,
        "grad_clip": 1e6,
        "clip_actions": False,
        "num_workers": args.n_workers,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,  
        "framework": "tf2",
        "seed": grid_search([(i + 1) * 100 for i in range(args.n_seeds)])
    }

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
        }
    }

    run_experiments(experiments, reuse_actors=True, concurrent=True)

    ray.shutdown()