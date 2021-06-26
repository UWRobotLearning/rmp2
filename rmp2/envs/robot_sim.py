"""
acceleration-based control for a pybullet robot
"""

from rmp2.utils.robot_config_utils import get_robot_urdf_path, get_robot_eef_uid
import numpy as np

# pybullet macro
JOINT_POSE_IDX = 0
JOINT_VEL_IDX = 1
JOINT_TORQUE_IDX = 3

JOINT_NAME_IDX = 1
JOINT_TYPE_IDX = 2
JOINT_LOWER_LIMIT_IDX = 8
JOINT_UPPER_LIMIT_IDX = 9
JOINT_VEL_LIMIT_IDX = 11
LINK_NAME_IDX = 12

# control modes:

# CLOSED LOOP (NOT RECOMMENDED): Both the actual
# joint angles and actual joint velocities are 
# used to compute the reference next-step joint 
# angles and velocities for the low-level pd 
# controller. This often leads to unstable behavior 
# and is hence not recommended
CLOSED_LOOP = 0
# VELOCITY OPEN LOOP (DEFAULT): The actual joint 
# angles and virtual joint velocities (computed 
# through numerically integrating the accelerations) 
# are used to compute the reference joint angles and 
# velocities
VEL_OPEN_LOOP = 1
# OPEN LOOP: The virtual joint angles and joint 
# velocities (both computed through numerical
# integration) are used to compute the reference 
# joint angles and velocities
OPEN_LOOP = 2

class RobotSim(object):
    """
    acceleration-based control for a pybullet robot
    """
    def __init__(self, urdf_path, eef_uid, bullet_client, time_step, mode=VEL_OPEN_LOOP):
        self.bullet_client = bullet_client
        self.time_step = time_step
        self.bullet_client.setTimeStep(time_step)

        assert mode == CLOSED_LOOP or mode == VEL_OPEN_LOOP or mode == OPEN_LOOP
        self._mode = mode

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.robot_uid = self.bullet_client.loadURDF(urdf_path, useFixedBase=True, flags=flags)

        self.joint_poses = None
        self.joint_vels = None
        self.joint_torques = None

        self.eef_uid = eef_uid

        self._joint_indices = None
        self._joint_lower_limit = None
        self._joint_upper_limit = None
        self._joint_vel_limit = None

        self._set_joint_indices()
        self.cspace_dim = len(self._joint_indices)


    def reset(self, initial_config, initial_vel):
        """
        reset the robot to initial configuration and velocity
        """
        initial_config = np.array(initial_config)
        initial_vel = np.array(initial_vel)

        for i, j in enumerate(self._joint_indices):
            self.bullet_client.resetJointState(self.robot_uid, j, 
            targetValue=initial_config[i],
            targetVelocity=initial_vel[i])

        self.joint_poses = np.array(initial_config)
        self.joint_vels = np.array(initial_vel)
        self.joint_torques = np.zeros_like(self.joint_poses)

        self.target_joint_poses = self.joint_poses
        self.target_joint_vels = self.joint_vels

    def step(self, action):
        """
        apply velocity control to the robot
        :param action: joint accelerations
        """
        if self.joint_poses is None or self.joint_vels is None:
            raise Exception('Error: make sure to call reset() before step!')
        
        # forward predict using Euler integration
        self.target_joint_poses = self.joint_poses + self.joint_vels * self.time_step
        self.target_joint_vels = self.joint_vels + action * self.time_step
        
        # clip to respect joint limits and joint velocity limits
        self.target_joint_poses = np.clip(self.target_joint_poses, self._joint_lower_limit, self._joint_upper_limit)
        self.target_joint_vels[self.target_joint_poses == self._joint_lower_limit] = 0.
        self.target_joint_vels[self.target_joint_poses == self._joint_upper_limit] = 0.
        self.target_joint_vels = np.clip(self.target_joint_vels, -self._joint_vel_limit, self._joint_vel_limit)
        
        self.bullet_client.setJointMotorControlArray(
            self.robot_uid, self._joint_indices, 
            self.bullet_client.POSITION_CONTROL, 
            targetPositions=self.target_joint_poses, targetVelocities=self.target_joint_vels)

    def get_observation(self):
        """
        joint angles and velocities of the robot.
        for CLOSED_LOOP (NOT RECOMMENDED): both joint 
            angles and velocities are given by the 
            pybullet simulator
        for VEL_OPEN_LOOP: joint angles are given by the
            pybullet simulator, yet the joint velocities 
            are given by numerical integration
        for OPEN_LOOP: both joint angles and velocities
            are given by numerical integration
        """
        full_joint_states = self.bullet_client.getJointStates(
            self.robot_uid, self._joint_indices)
        if self._mode == OPEN_LOOP:
            self.joint_poses = self.target_joint_poses
            self.joint_vels = self.target_joint_vels
        elif self._mode == VEL_OPEN_LOOP:
            self.joint_poses = np.array([joint_state[JOINT_POSE_IDX] for joint_state in full_joint_states])
            self.joint_vels = self.target_joint_vels
        elif self._mode == CLOSED_LOOP:
            self.joint_poses = np.array([joint_state[JOINT_POSE_IDX] for joint_state in full_joint_states])
            self.joint_vels = np.array([joint_state[JOINT_VEL_IDX] for joint_state in full_joint_states])
        self.joint_torques = np.array([joint_state[JOINT_TORQUE_IDX] for joint_state in full_joint_states])
        return self.joint_poses.copy(), self.joint_vels.copy(), self.joint_torques.copy()

    def _set_joint_indices(self):
        """
        set the joint limits for the robot
        """
        self._joint_indices = []
        self._joint_lower_limit = []
        self._joint_upper_limit = []
        self._joint_vel_limit = []
        self._link_index = {}

        for j in range(self.bullet_client.getNumJoints(self.robot_uid)):
            info = self.bullet_client.getJointInfo(self.robot_uid, j)
            joint_name = info[JOINT_NAME_IDX]
            joint_type = info[JOINT_TYPE_IDX]
            joint_lower_limit = info[JOINT_LOWER_LIMIT_IDX]
            joint_upper_limit = info[JOINT_UPPER_LIMIT_IDX]
            joint_vel_limit = info[JOINT_VEL_LIMIT_IDX]
            link_name = info[LINK_NAME_IDX]

            self._link_index[link_name.decode("utf-8")] = j
        
            if (joint_type == self.bullet_client.JOINT_PRISMATIC or joint_type == self.bullet_client.JOINT_REVOLUTE):
                self._joint_indices.append(j)
                self._joint_lower_limit.append(joint_lower_limit)
                self._joint_upper_limit.append(joint_upper_limit)
                self._joint_vel_limit.append(joint_vel_limit)

        self._joint_lower_limit = np.array(self._joint_lower_limit)
        self._joint_upper_limit = np.array(self._joint_upper_limit)
        self._joint_vel_limit = np.array(self._joint_vel_limit)

        indices = self._joint_lower_limit > self._joint_upper_limit
        self._joint_lower_limit[indices] = -np.inf
        self._joint_upper_limit[indices] = np.inf
        indices = self._joint_vel_limit <= 0
        self._joint_vel_limit[indices] = np.inf


def create_robot_sim(robot_name, bullet_client, time_step, mode=VEL_OPEN_LOOP):
    """
    create a acceleration-based control robot given name
    :param robot_name: robot name, 3link or franka
    :param bullet_client: pybullet client
    :param time_step: simulation time between steps
    :param mode: control mode (see macros)
    :return robot_sim: RobotSim object for 
    acceleration-based control of the robot
    """
    urdf_path = get_robot_urdf_path(robot_name)
    eef_uid = get_robot_eef_uid(robot_name)
    robot_sim = RobotSim(urdf_path, eef_uid, bullet_client=bullet_client, time_step=time_step, mode=mode)
    return robot_sim

