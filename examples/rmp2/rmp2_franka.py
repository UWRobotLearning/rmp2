"""
Script for rolling out a hand-designed rmp2 policy on the franka robot
rmp parameters are given in rmp2/configs/franka_config.yaml
"""

from rmp2.rmpgraph import RobotRMPGraph
from rmp2.envs import FrankaEnv
from rmp2.utils.env_wrappers import FrankaFullRMPWrapper
import tensorflow as tf

n_trials = 10
seed = 15
dtype = "float32"

env_wrapper = FrankaFullRMPWrapper(dtype=dtype)
rmp_graph = RobotRMPGraph(robot_name="franka", dtype=dtype)

config = {
    "goal": [0.5, -0.5, 0.5],
    "horizon": 1800,
    "action_repeat": 3,
    "q_init": [ 0.0000, -0.7854,  0.0000, -2.4435,  0.0000,  1.6581,  0.75],
    "render": True,
}

goal = tf.convert_to_tensor([config['goal']])

def policy(state):
    ts_state = tf.convert_to_tensor([state])
    policy_input = env_wrapper.obs_to_policy_input(ts_state)
    policy_input['goal'] = goal
    ts_action = rmp_graph(**policy_input)
    action = ts_action[0].numpy()
    return action

env = FrankaEnv(config)
env.seed(seed)
state = env.reset()
action = policy(state)

for _ in range(n_trials):
    state = env.reset()
    while True:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        if done:
            break

