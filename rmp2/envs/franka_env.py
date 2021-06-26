"""
Base gym environment for franka robot
"""

from rmp2.envs.robot_env import RobotEnv
from rmp2.utils.np_utils import sample_from_torus_3d
from rmp2.utils.python_utils import merge_dicts
from rmp2.utils.bullet_utils import add_goal, add_obstacle_ball
import numpy as np

DEFAULT_CONFIG = {
    # parameters for randomly generated goals
    "goal_torus_angle_center": 0., 
    "goal_torus_angle_range": np.pi,
    "goal_torus_major_radius": 0.5,
    "goal_torus_minor_radius": 0.3,
    "goal_torus_height": 0.5,
    # parameters for randomly generated obstacles
    "obs_torus_angle_center": 0., 
    "obs_torus_angle_range": np.pi,
    "obs_torus_major_radius": 0.5,
    "obs_torus_minor_radius": 0.3,
    "obs_torus_height": 0.5,
    # obstacle size
    "max_obstacle_radius": 0.1,
    "min_obstacle_radius": 0.05,
    # init min goal distance
    "initial_goal_distance_min": 0.5, 

}

class FrankaEnv(RobotEnv):
    """
    Base gym environment for franka robot
    """
    def __init__(self, config=None):
        if config is not None:
            config = merge_dicts(DEFAULT_CONFIG, config)
        else:
            config = DEFAULT_CONFIG.copy()

        # random goal config
        self._goal_torus_angle_center = config["goal_torus_angle_center"]
        self._goal_torus_angle_range = config["goal_torus_angle_range"]
        self._goal_torus_major_radius = config["goal_torus_major_radius"]
        self._goal_torus_minor_radius = config["goal_torus_minor_radius"]
        self._goal_torus_height = config["goal_torus_height"]
        # random obstacle config
        self._obs_torus_angle_center = config["obs_torus_angle_center"]
        self._obs_torus_angle_range = config["obs_torus_angle_range"]
        self._obs_torus_major_radius = config["obs_torus_major_radius"]
        self._obs_torus_minor_radius = config["obs_torus_minor_radius"]
        self._obs_torus_height = config["obs_torus_height"]

        super().__init__(
            robot_name="franka",
            workspace_dim=3,
            config=config)

    def _generate_random_goal(self):
        # if goal is given, use the fixed goal
        if self.goal is None:
            current_goal = sample_from_torus_3d(
                self.np_random,
                self._goal_torus_angle_center, 
                self._goal_torus_angle_range,
                self._goal_torus_major_radius,
                self._goal_torus_minor_radius,
                self._goal_torus_height)
        # otherwise, sample a random goal with the specified parameters
        else:
            current_goal = self.goal
        # generate goal object within pybullet
        goal_uid = add_goal(self._p, current_goal)
        return current_goal, goal_uid
        
    def _generate_random_obstacles(self):
        current_obs = []
        obs_uids = []

        # if obstacle config list is given, sample one config from the list
        if self.obstacle_cofigs is not None:
            config = self.obstacle_cofigs[self.np_random.randint(len(self.obstacle_cofigs))]
            for (i, obstacle) in enumerate(config):
                obs_uids.append(
                    add_obstacle_ball(self._p, obstacle['center'], obstacle['radius'])
                )
                current_obs.append(np.append(obstacle['center'], obstacle['radius']))
            for i in range(len(config), self.max_obstacle_num):
                current_obs.append(np.append(np.zeros(self.workspace_dim), -1.))
        # otherwise, sample random obstacles with the specified parameters
        else:
            num_obstacles = self.np_random.randint(self.min_obstacle_num, self.max_obstacle_num + 1)
            for i in range(self.max_obstacle_num):
                if i < num_obstacles:
                    radius = self.np_random.uniform(low=self.min_obstacle_radius, high=self.max_obstacle_radius)
                    center = sample_from_torus_3d(
                        self.np_random,
                        self._obs_torus_angle_center, 
                        self._obs_torus_angle_range,
                        self._obs_torus_major_radius,
                        self._obs_torus_minor_radius,
                        self._obs_torus_height)
                    obs_uids.append(
                        add_obstacle_ball(self._p, center, radius)
                    )
                    current_obs.append(np.append(center, radius))
                else:
                    current_obs.append(np.append(np.zeros(self.workspace_dim), -1.))
        # generate obstacle objects within pybullet
        current_obs = np.array(current_obs).flatten()
        return current_obs, obs_uids