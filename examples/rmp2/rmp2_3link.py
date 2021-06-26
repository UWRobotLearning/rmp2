from rmp2.rmpgraph import RobotRMPGraph
from rmp2.envs import ThreeLinkEnv
from rmp2.utils.env_wrappers import ThreeLinkFullRMPWrapper
import tensorflow as tf

n_trials = 10
seed = 15
dtype = "float32"

env_wrapper = ThreeLinkFullRMPWrapper(dtype=dtype)
rmp_graph = RobotRMPGraph(robot_name="3link", workspace_dim=2, dtype=dtype)

config = {
    "goal": [-0.5, 0.1],
    "horizon": 1800,
    "action_repeat": 3,
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

env = ThreeLinkEnv(config)
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

