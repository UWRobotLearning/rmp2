from rmp2.envs.franka_env import FrankaEnv
from rmp2.rmpgraph.robotics import RobotRMPGraph
from rmp2.utils.python_utils import merge_dicts
import tensorflow as tf
import numpy as np
from gym import spaces
import os

BULLET_LINK_POSE_INDEX = 4
BULLET_LINK_VEL_INDEX = 6

DEFAULT_CONFIG = {
    "residual": True,
    "dtype": "float32",
    "offset": 1e-3,
}

class FrankaResidualRMPEnv(FrankaEnv):
    def __init__(self, config=None):
        if config is not None:
            config = merge_dicts(DEFAULT_CONFIG, config)
        else:
            config = DEFAULT_CONFIG.copy()
        
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '../configs/franka_residual_rmp_config.yaml'
            )

        self.rmp_graph = RobotRMPGraph(
            'franka',
            config_path=config_path, 
            dtype=config['dtype'], 
            offset=config['offset'])

        self.dtype = config['dtype']
        self._ts_goal = None
        self._ts_obs = None

        self.external_rmp_names = []
        self.external_rmp_links = []
        for key, rmp in zip(self.rmp_graph.keys, self.rmp_graph.rmps):
            if key == 'external_rmp':
                self.external_rmp_names.append(rmp.feature_name)
                self.external_rmp_links.append(rmp.link)

        super().__init__(config=config)

        action_space_dim = 0
        for link in self.external_rmp_links:
            if link == 'joint':
                action_space_dim += self.cspace_dim * (self.cspace_dim + 1)
            else:
                action_space_dim += self.workspace_dim * (self.workspace_dim + 1)
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(action_space_dim,), dtype=self.dtype)


    def _generate_random_goal(self):
        current_goal, goal_uid = super()._generate_random_goal()
        self._ts_goal = tf.convert_to_tensor([current_goal], dtype=self.dtype)
        return current_goal, goal_uid

    def _generate_random_obstacles(self):
        current_obs, obs_uids = super()._generate_random_obstacles()
        self._ts_obs = tf.convert_to_tensor([current_obs], dtype=self.dtype)
        self._ts_obs = tf.reshape(self._ts_obs, (1, -1, self.workspace_dim + 1))
        return current_obs, obs_uids

    def step(self, external_rmps):
        index = 0
        features = {}
        for name, link in zip(self.external_rmp_names, self.external_rmp_links):
            if link == 'joint':
                dim = self.cspace_dim
            else:
                dim = self.workspace_dim
            metric_sqrt = external_rmps[index: index + dim ** 2]
            metric_sqrt = np.reshape(metric_sqrt, (dim, dim))
            index += dim ** 2
            accel = external_rmps[index: index + dim]
            index += dim

            ts_metric_sqrt = tf.convert_to_tensor([metric_sqrt], dtype=self.dtype)
            ts_accel = tf.convert_to_tensor([accel], dtype=self.dtype)
            features[name] = (ts_metric_sqrt, ts_accel)

        joint_poses, joint_vels, _ = self._robot.get_observation()
        ts_joint_poses = tf.convert_to_tensor([joint_poses], dtype=self.dtype)
        ts_joint_vels = tf.convert_to_tensor([joint_vels], dtype=self.dtype)
        
        ts_action = self.rmp_graph(
            ts_joint_poses, ts_joint_vels, 
            obstacles=self._ts_obs, goal=self._ts_goal,
            **features)
        action = ts_action.numpy()
        action = action.flatten()

        return super().step(action)

    def get_extended_observation(self):
        link_poses = []
        link_vels = []

        for link in self.external_rmp_links:
            link_state = self._p.getLinkState(
                self._robot.robot_uid, self._robot._link_index[link], 
                computeLinkVelocity=1,
                computeForwardKinematics=1)
            link_pose = link_state[BULLET_LINK_POSE_INDEX][:self.workspace_dim]
            link_vel = link_state[BULLET_LINK_VEL_INDEX][:self.workspace_dim]
            link_poses.append(link_pose)
            link_vels.append(link_vel)
        link_poses = np.array(link_poses).flatten()
        link_vels = np.array(link_vels).flatten()

        observation = super().get_extended_observation()

        self._observation = np.concatenate(
            (observation, self.current_goal, link_poses, link_vels)
        )

        return self._observation