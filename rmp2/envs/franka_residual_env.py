from rmp2.envs.franka_env import FrankaEnv
from rmp2.rmpgraph.robotics import RobotRMPGraph
from rmp2.utils.python_utils import merge_dicts
import tensorflow as tf
import numpy as np
import os

DEFAULT_CONFIG = {
    "dtype": "float32",
    "offset": 1e-3,
}

class FrankaResidualEnv(FrankaEnv):
    def __init__(self, config=None):
        if config is not None:
            config = merge_dicts(DEFAULT_CONFIG, config)
        else:
            config = DEFAULT_CONFIG.copy()

        # TODO: change config file
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '../configs/franka_residual_config.yaml'
            )

        self.rmp_graph = RobotRMPGraph(
            'franka',
            config_path=config_path, 
            dtype=config['dtype'], 
            offset=config['offset'])

        self.dtype = config['dtype']
        self._ts_goal = None
        self._ts_obs = None

        super().__init__(config=config)

    def _generate_random_goal(self):
        current_goal, goal_uid = super()._generate_random_goal()
        self._ts_goal = tf.convert_to_tensor(np.array([current_goal]), dtype=self.dtype)
        return current_goal, goal_uid

    def _generate_random_obstacles(self):
        current_obs, obs_uids = super()._generate_random_obstacles()
        self._ts_obs = tf.convert_to_tensor(np.array([current_obs]), dtype=self.dtype)
        self._ts_obs = tf.reshape(self._ts_obs, (1, -1, self.workspace_dim + 1))
        return current_obs, obs_uids

    def step(self, residual_action):
        joint_poses, joint_vels, _ = self._robot.get_observation()
        ts_joint_poses = tf.convert_to_tensor([joint_poses], dtype=self.dtype)
        ts_joint_vels = tf.convert_to_tensor([joint_vels], dtype=self.dtype)
        ts_action = self.rmp_graph(ts_joint_poses, ts_joint_vels, obstacles=self._ts_obs, goal=self._ts_goal)
        action = ts_action.numpy()
        action = action.flatten()
        action = np.clip(action, self._action_space.low, self._action_space.high)
        action = action + residual_action
        return super().step(action)
