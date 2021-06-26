"""
Computing the robot forword kinematics in tensorflow
"""

from rmp2.utils.tf_transform_utils import T_prismatic, T_revolute, T_rpy
from urdf_parser_py.urdf import URDF
import tensorflow as tf
import os


class Joint:
    """
    tensorflow module for joint
    """
    def __init__(self, urdf_joint, dtype=tf.float32):
        """
        :param urdf_joint: joint class from urdf_parser_py
        :param dtype: data type
        """
        self.name = urdf_joint.name
        self.parent = urdf_joint.parent
        self.child = urdf_joint.child
        self.type = urdf_joint.type

        if urdf_joint.limit is not None:
            self.lower_limit = urdf_joint.limit.lower
            self.upper_limit = urdf_joint.limit.upper
            self.velocity_limit = urdf_joint.limit.velocity

        if urdf_joint.axis is None:
            self.axis = tf.constant([1., 0., 0.], dtype=dtype)
        else:
            axis = tf.convert_to_tensor(urdf_joint.axis, dtype=dtype)
            axis = (1. / tf.linalg.norm(axis)) * axis
            self.axis = axis

        self.xyz = tf.constant([0., 0., 0.], dtype=dtype)
        self.rpy = tf.constant([0., 0., 0.], dtype=dtype)
        if urdf_joint.origin is not None:
            if urdf_joint.origin.xyz is not None:
                self.xyz = tf.convert_to_tensor(urdf_joint.origin.xyz, dtype=dtype)
            if urdf_joint.origin.rpy is not None:
                self.rpy = tf.convert_to_tensor(urdf_joint.origin.rpy, dtype=dtype)

    def transformation(self, q=None):
        """
        homogeneous transformation matrices for the joint
        :param q: joint configuration
        :return frame: homogenous transformation matrices
        """
        if self.type == "fixed":
            assert q is None
        else:
            assert q is not None
        if self.type == "fixed":
            # todo: batch_size is not given here!
            frame = T_rpy(self.xyz, self.rpy)
        elif self.type == "prismatic":
            frame = T_prismatic(self.xyz, self.rpy, self.axis, q)
        elif self.type in ["revolute", "continuous"]:
            frame = T_revolute(self.xyz, self.rpy, self.axis, q)
        return frame


class Robot:
    """
    tensorflow module for robot forward kinematics
    """
    def __init__(self, urdf_path, link_set=None, workspace_dim=3, dtype=tf.float32):
        """
        :param urdf_path: path to urdf file
        :param link_set: list of links to compute fk
        :param workspace_dim: workspace dimension
        :param dtype: data type
        """
        self.workspace_dim = workspace_dim
        self.dtype = dtype

        urdf_robot = URDF.from_xml_file(file_path=urdf_path)
        self.parent_map = urdf_robot.parent_map
        self.child_map = urdf_robot.child_map

        self.base_link = urdf_robot.get_root()
        if link_set is not None:
            self.link_set = list(link_set)
        else:
            self.link_set = list(urdf_robot.link_map.keys())

        # only take the joints that are required to compute fk for links in link_set
        self.joint_map = {}
        for link in self.link_set:
            chain = urdf_robot.get_chain(self.base_link, link, links=False)
            for joint in chain:
                if not joint in self.joint_map:
                    self.joint_map[joint] = Joint(urdf_robot.joint_map[joint], dtype=dtype)
        # topologically sort the joints
        self.set_joint_names()

        self.joint_lower_limits = tf.constant([self.joint_map[joint].lower_limit for joint in self.non_fixed_joint_names], dtype=self.dtype)
        self.joint_upper_limits = tf.constant([self.joint_map[joint].upper_limit for joint in self.non_fixed_joint_names], dtype=self.dtype)
        self.joint_velocity_limits = tf.constant([self.joint_map[joint].velocity_limit for joint in self.non_fixed_joint_names], dtype=self.dtype)


    def forward_kinematics(self, q):
        """
        compute the forward kinematics
        ---------------------------
        :param q: configuration space coordinate
        :return x: dictionary of task space coordinates
        """
        if len(q.shape) == 1:
            q = tf.expand_dims(q, 0)
        batch_size = q.shape[0]

        frames = {}
        frame = tf.eye(4, batch_shape=(batch_size,), dtype=self.dtype)
        frames[self.base_link] = frame
        # recursively compute fk
        self._forward_kinematics(self.base_link, q, frames)

        # take the coordinates corresponding to the links in link_set
        x = {}
        for link in self.link_set:
            x[link] = frames[link][:, 0:self.workspace_dim, -1]
        return x

    def _forward_kinematics(self, base_link, q, frames):
        """
        recursively compute forward kinematics
        ------------------------
        :param base_link: the name of the current base link
        :param q: joint coordinate from base_link upwards
        :param frames: homogenous transformation from configuration space to base_link space
        """
        assert base_link in frames

        # base_link is a leaf in the kinematic tree
        if base_link not in self.child_map:
            return

        # compute fk for the child links
        children_list = self.child_map[base_link]
        for child_joint_name, child_link in children_list:
            if child_joint_name in self.joint_map:
                child_joint = self.joint_map[child_joint_name]
                if child_joint.type == 'fixed':
                    frames[child_link] = tf.matmul(frames[base_link], child_joint.transformation())
                    self._forward_kinematics(child_link, q, frames)
                else:
                    frames[child_link] = tf.matmul(frames[base_link], child_joint.transformation(q[:, 0]))
                    q = q[:, 1:]
                    self._forward_kinematics(child_link, q, frames)


    def set_joint_names(self):
        """
        topologically sort the joint names
        """
        joint_names = []
        non_fixed_joint_names = []
        self._sort_joint_names(self.base_link, joint_names, non_fixed_joint_names)
        self.joint_names = joint_names
        self.non_fixed_joint_names = non_fixed_joint_names


    def _sort_joint_names(self, base_link, joint_names, non_fixed_joint_names):
        """
        sort the joint names recursively
        """
        # base_link is a leaf in the kinematic tree
        if base_link not in self.child_map:
            return

        children_list = self.child_map[base_link]
        for child_joint_name, child_link in children_list:
            if child_joint_name in self.joint_map:
                child_joint = self.joint_map[child_joint_name]
                joint_names.append(child_joint_name)
                if child_joint.type != 'fixed':
                    non_fixed_joint_names.append(child_joint_name)
                self._sort_joint_names(child_link, joint_names, non_fixed_joint_names)

    @property
    def num_joints(self):
        return len(self.non_fixed_joint_names)

    @property
    def cspace_dim(self):
        return len(self.non_fixed_joint_names)