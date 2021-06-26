from rmp2.envs import robot_sim
from rmp2.utils.python_utils import merge_dicts
from rmp2.utils.bullet_utils import add_collision_goal
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import numpy as np
import time
from pkg_resources import parse_version
from abc import abstractmethod

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

BULLET_LINK_POSE_INDEX = 4
BULLET_CLOSEST_POINT_CONTACT_FLAG_INDEX = 0
BULLET_CLOSEST_POINT_CONTACT_NORMAL_INDEX = 7
BULLET_CLOSEST_POINT_DISTANCE_INDEX = 8

DEFAULT_CONFIG = {
    # time setup
    "time_step": 1/240.,
    "action_repeat": 3,
    "horizon": 1200,
    "terminate_after_collision": True,
    # workspace radius
    "workspace_radius": 1.,
    # goal setup
    "goal": None,
    # initial config setup
    "q_init": None,
    # obstacle setups
    "obstacle_configs": None,
    "max_obstacle_num": 3,
    "min_obstacle_num": 3,
    "max_obstacle_radius": 0.15,
    "min_obstacle_radius": 0.05,
    # initialization buffer btw obstacles and robot & goal
    "initial_collision_buffer": 0.1,
    "initial_joint_limit_buffer": 0.1,
    "initial_goal_distance_min": 0.0, 
    # reward parameters
    "goal_reward_model": "gaussian",
    "goal_reward_length_scale": 0.1,
    "obs_reward_model": "linear",
    "obs_reward_length_scale": 0.05,
    # "collision_cost": 1000.,
    "goal_reward_weight": 1.,
    "obs_reward_weight": 1.,
    "ctrl_reward_weight": 1e-5,
    "max_reward": 5.,
    # actuation setup
    "actuation_limit": 1.,
    # render setup
    "render": False,
    "cam_dist": 1.5,
    "cam_yaw": 30,
    "cam_pitch": -45,
    "cam_position": [0, 0, 0],
    # pybullet gravity
    "gravity": -9.8,
    # acceleration control mode
    "acc_control_mode": robot_sim.VEL_OPEN_LOOP,
}


class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                robot_name,
                workspace_dim,
                config=None):
        # merge config with default config
        if config is not None:
            config = merge_dicts(DEFAULT_CONFIG, config)
        else:
            config = DEFAULT_CONFIG.copy()

        self.robot_name = robot_name

        # time setups
        self._time_step = config['time_step']
        self._action_repeat = config['action_repeat']
        self._horizon = config['horizon']
        self._terminate_after_collision = config['terminate_after_collision']
        self._env_step_counter = 0
        self.terminated = False

        # render setups
        self._render = config['render']
        self._cam_dist = config['cam_dist']
        self._cam_yaw = config['cam_yaw']
        self._cam_pitch = config['cam_pitch']
        self._cam_position = config['cam_position']
        self._gravity = config['gravity']

        # connect to pybullet environment
        self._p = p
        if self._render:
            cid = self._p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = self._p.connect(p.GUI)
                self._p.resetDebugVisualizerCamera(
                    cameraDistance=self._cam_dist, 
                    cameraYaw=self._cam_yaw, 
                    cameraPitch=self._cam_pitch, 
                    cameraTargetPosition=self._cam_position)
        else:
            self._p.connect(p.DIRECT)
        
        # create the robot in pybullet
        self._robot = robot_sim.create_robot_sim(self.robot_name, self._p, self._time_step)
        self.cspace_dim = self._robot.cspace_dim
        self.workspace_dim = workspace_dim
        self.workspace_radius = config["workspace_radius"]
        
        # set up goal
        self.goal = config["goal"]
        self.current_goal = None
        self.goal_uid = None

        # set up initial config
        self.q_init = config["q_init"]

        # set up obstacle config
        self.obstacle_cofigs = config["obstacle_configs"]
        if self.obstacle_cofigs is None:
            self.max_obstacle_num = config["max_obstacle_num"]
            self.min_obstacle_num = config["min_obstacle_num"]
            self.max_obstacle_radius = config["max_obstacle_radius"]
            self.min_obstacle_radius = config["min_obstacle_radius"]
        else:
            self.max_obstacle_num = max(len(c) for c in self.obstacle_cofigs)
        self.current_obs = []
        self.obs_uids = []

        # set up rewards
        self._goal_reward_weight = config["goal_reward_weight"]
        self._obs_reward_weight = config["obs_reward_weight"]
        self._ctrl_reward_weight = config["ctrl_reward_weight"]
        self._goal_reward_model = config["goal_reward_model"]
        self._goal_reward_length_scale = config["goal_reward_length_scale"]
        self._obs_reward_model = config["obs_reward_model"]
        self._obs_reward_length_scale = config["obs_reward_length_scale"]
        self._max_reward = config["max_reward"]

        # initialization parameters
        self._initial_collision_buffer = config["initial_collision_buffer"]
        self._initial_joint_limit_buffer = config["initial_joint_limit_buffer"]
        self._initial_goal_distance_min = config["initial_goal_distance_min"]

        self._acc_control_mode = config["acc_control_mode"]

        # set up random seed
        self.seed()

        # set up action space and observation space
        self.action_space = spaces.Box(
            low=-config["actuation_limit"], high=config["actuation_limit"], 
            shape=(self.cspace_dim,), dtype=np.float32)
        self._action_space = spaces.Box(
            low=-config["actuation_limit"], high=config["actuation_limit"], 
            shape=(self.cspace_dim,), dtype=np.float32)
        self._observation = self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=self._observation.shape, dtype=np.float32)

        self.viewer = None

    def reset(self):
        # reset time and simulator
        self.terminated = False
        self._env_step_counter = 0
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        self._p.setTimeStep(self._time_step)
        self._p.setGravity(0, 0, self._gravity)

        self.goal_uid = None
        self.obs_uids = []

        # create robot
        self._robot = robot_sim.create_robot_sim(self.robot_name, self._p, self._time_step, mode=self._acc_control_mode)

        # keep generating initial configurations until a valid one
        while True:
            self._clear_goal_and_obstacles()
            self._generate_random_initial_config()
            self.current_goal, self.goal_uid = self._generate_random_goal()
            self.current_obs, self.obs_uids = self._generate_random_obstacles()
            self._p.stepSimulation()

            # check if the initial configuration is valid
            if self.goal is None:
                eef_position = np.array(self._p.getLinkState(
                    self._robot.robot_uid, 
                    self._robot.eef_uid)[BULLET_LINK_POSE_INDEX])
                distance_to_goal = np.linalg.norm(
                    eef_position[:self.workspace_dim] - self.current_goal[:self.workspace_dim])
            else:
                distance_to_goal = np.inf

            if not self._collision(buffer=self._initial_collision_buffer) and \
                not self._goal_obstacle_collision(buffer=self._initial_collision_buffer) and \
                distance_to_goal >= self._initial_goal_distance_min:
                print('Successfully generated a valid initial configuration')
                break
            print('config in collision...regenerating...')
        self._observation = self.get_extended_observation()
        return np.array(self._observation)

    def __del__(self):
        self._p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_extended_observation(self):
        joint_poses, joint_vels, _ = self._robot.get_observation()

        # vector eef to goal
        eef_position = np.array(self._p.getLinkState(self._robot.robot_uid, self._robot.eef_uid)[BULLET_LINK_POSE_INDEX])
        delta_x = self.current_goal[:self.workspace_dim] - eef_position[:self.workspace_dim]

        # vector to closest point on obstacles
        vector_obstacles = []
        for obs_uid in self.obs_uids:
            closest_points = self._p.getClosestPoints(self._robot.robot_uid, obs_uid, 1000)
            distance_to_obs = 1000.
            vector_obs = np.array([0., 0., 0.])
            for point in closest_points:
                if point[BULLET_CLOSEST_POINT_DISTANCE_INDEX] <= distance_to_obs:
                    distance_to_obs = point[BULLET_CLOSEST_POINT_DISTANCE_INDEX]
                    vector_obs = distance_to_obs * np.array(point[BULLET_CLOSEST_POINT_CONTACT_NORMAL_INDEX])
            vector_obstacles.append(vector_obs[:self.workspace_dim])
        for i in range(len(self.obs_uids), self.max_obstacle_num):
            vector_obstacles.append(np.zeros((self.workspace_dim,)))
        
        vector_obstacles = np.array(vector_obstacles).flatten()

        self._observation = np.concatenate(
            (np.sin(joint_poses),
            np.cos(joint_poses),
            joint_vels,
            delta_x,
            vector_obstacles,
            self.current_obs)
        )
        return self._observation


    def step(self, action):
        action = np.clip(action, self._action_space.low, self._action_space.high)
        action[np.isnan(action)] = 0.
        _reward = 0
        done = False
        for i in range(self._action_repeat):
            time_start = time.time()
            self._robot.step(action)
            self._p.stepSimulation()
            _reward += self._get_reward()

            # if render, sleep for appropriate duration
            if self._render:
                computation_time = time.time() - time_start
                time.sleep(max(self._time_step - computation_time, 0.))
            
            # check if terminated
            if self._termination():
                done = True
                break
            self._env_step_counter += 1
        
        reward = _reward / (i + 1)
            
        self._observation = self.get_extended_observation()

        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._robot.robot_uid)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                        nearVal=0.1,
                                                        farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                    height=RENDER_HEIGHT,
                                                    viewMatrix=view_matrix,
                                                    projectionMatrix=proj_matrix,
                                                    renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        if (self.terminated or self._env_step_counter > self._horizon):
            self._observation = self.get_extended_observation()
            self.terminated = True
            return True
        
        # check collision only if terminates after collision
        if not self._terminate_after_collision:
            return False
        
        # check collision with obstacles
        if self._collision():
            self.terminated = True
            return True
        
        return False

    def _get_reward(self):
        # rewards for goal
        eef_position = np.array(self._p.getLinkState(self._robot.robot_uid, self._robot.eef_uid)[BULLET_LINK_POSE_INDEX])
        distance_to_goal = np.linalg.norm(eef_position[:self.workspace_dim] - self.current_goal[:self.workspace_dim])
        reward_goal = self._get_goal_reward(distance_to_goal)

        # rewards for obstacles
        reward_obs = 0.
        for obs_uid in self.obs_uids:
            closest_points = self._p.getClosestPoints(self._robot.robot_uid, obs_uid, 1000)
            distance_to_obs = 1000.
            for point in closest_points:
                distance_to_obs = min(distance_to_obs, point[BULLET_CLOSEST_POINT_DISTANCE_INDEX])
                # if in collision, set terminated to True
                if point[BULLET_CLOSEST_POINT_CONTACT_FLAG_INDEX] and self._terminate_after_collision:
                    self.terminated = True
            reward_obs += self._get_obstacle_reward(distance_to_obs)

        # TODO: maybe fix the interface for getting joint torques one day
        _, _, joint_torques = self._robot.get_observation()
        reward_ctrl = - np.square(joint_torques).sum()

        reward = reward_goal * self._goal_reward_weight + \
                reward_obs * self._obs_reward_weight + \
                reward_ctrl * self._ctrl_reward_weight
        reward = np.clip(reward, -self._max_reward, self._max_reward)
        return reward

    def _get_obstacle_reward(self, distance_to_obs):
        if self._obs_reward_model == "linear":
            reward_obs = - max(0., 1. - 1. * distance_to_obs / self._obs_reward_length_scale)
        elif self._obs_reward_model == "gaussian":
            reward_obs = - np.exp(-0.5 * distance_to_obs ** 2 / self._obs_reward_length_scale ** 2)
        elif self._obs_reward_model == "laplace":
            reward_obs = - np.exp(-distance_to_obs / self._obs_reward_length_scale)
        else:
            Warning('warning: invalid reward model')
            reward_obs = 0.
        return reward_obs

    def _get_goal_reward(self, distance_to_goal):
        if self._goal_reward_model == "linear":
            reward_goal = (self.workspace_radius * 2. - distance_to_goal) / self.workspace_radius / 2.
        elif self._goal_reward_model == "gaussian":
            reward_goal = np.exp(-0.5 * distance_to_goal ** 2 / self._goal_reward_length_scale ** 2)
        elif self._goal_reward_model == "laplace":
            reward_goal = np.exp(-distance_to_goal / self._goal_reward_length_scale)
        else:
            Warning('warning: invalid reward model')
            reward_goal = 0.
        return reward_goal


    def _collision(self, buffer=0.0):
        for obs_uid in self.obs_uids:
            closest_points = self._p.getClosestPoints(self._robot.robot_uid, obs_uid, buffer)
            if len(closest_points) > 0:
                return True
        
        # TODO: check self-collision, disabled for now
        if False:
            closest_points = self._p.getClosestPoints(self._robot.robot_uid, self._robot.robot_uid, buffer)
            # n_links = p.getNumJoints(self._robot.robot_uid) - 1
            if len(closest_points) > 31: # 3 * n_links - 2:
                return True
        return False

    def _goal_obstacle_collision(self, buffer=0.0):
        goal_position, _ = self._p.getBasePositionAndOrientation(self.goal_uid)
        collision_goal = add_collision_goal(self._p, goal_position)
        for obs_uid in self.obs_uids:
            closest_points = self._p.getClosestPoints(collision_goal, obs_uid, buffer)
            if len(closest_points) > 0:
                self._p.removeBody(collision_goal)
                return True
        self._p.removeBody(collision_goal)
        return False

    def _clear_goal_and_obstacles(self):
        # clear previous goals
        self.current_goal = None
        if self.goal_uid is not None:
            self._p.removeBody(self.goal_uid)
            self.goal_uid = None
        # clear previous obstacles
        self.current_obs = []
        for obs_uid in self.obs_uids:
            self._p.removeBody(obs_uid)
        self.obs_uids = []


    def _generate_random_initial_config(self):
        lower_limit = self._robot._joint_lower_limit
        upper_limit = self._robot._joint_upper_limit
        lower_limit = np.maximum(lower_limit + self._initial_joint_limit_buffer, -np.pi)
        upper_limit = np.minimum(upper_limit - self._initial_joint_limit_buffer, np.pi)
        if self.q_init is None:
            initial_config = self.np_random.uniform(low=lower_limit, high=upper_limit)
        else:
            initial_config = self.q_init + self.np_random.uniform(low=-0.1, high=0.1, size=self.cspace_dim)
            initial_config = np.clip(initial_config, lower_limit, upper_limit)
        initial_vel = self.np_random.uniform(low=-0.005, high=0.005, size=self.cspace_dim)
        # initial_vel = np.zeros_like(initial_config)
        self._robot.reset(initial_config, initial_vel)

    @abstractmethod
    def _generate_random_goal(self):
        pass
        
    @abstractmethod
    def _generate_random_obstacles(self):
        pass



    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step