from abc import ABC, abstractmethod
import tensorflow as tf


def load_env_wrapper(envid, dtype=tf.float32):
    if envid == '3link' or envid == '3link_residual':
        return ThreeLinkWrapper(dtype=dtype)
    elif envid == '3link_rmp':
        return ThreeLinkRMPWrapper(dtype=dtype)
    elif envid == 'franka' or envid == 'franka_residual':
        return FrankaWrapper(dtype=dtype)
    elif envid == 'franka_rmp':
        return FrankaRMPWrapper(dtype=dtype)
    else:
        raise ValueError


class EnvWrapper(ABC):
    @abstractmethod
    def obs_to_policy_input(self, obs):
        """
        params: environment observation
        return: input to the policy
        """
        
    
    @abstractmethod
    def obs_to_value_input(self, obs):
        """
        params: environment observation
        return: input to the value function
        """

    @abstractmethod
    def policy_output_to_action(self, actions):
        """
        params: output of the policy
        return: action to the environment
        """



class ThreeLinkWrapper(EnvWrapper):
    def __init__(self, dtype=tf.float32):
        self.dtype = dtype

    def obs_to_policy_input(self, obs):
        return tf.cast(obs, self.dtype)

    def obs_to_value_input(self, obs):
        return tf.cast(obs, self.dtype)

    @staticmethod
    def policy_output_to_action(actions):
        return actions


class ThreeLinkRMPWrapper(EnvWrapper):
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def obs_to_policy_input(self, obs):
        obs = tf.cast(obs, self.dtype)
        batch_size, obs_dim = obs.shape
        batch_size, obs_dim = int(batch_size), int(obs_dim)

        num_obstacles = int((obs_dim - 17) / 5)

        x = tf.gather(obs, range(obs_dim - 4, obs_dim - 2), axis=1)
        xd = tf.gather(obs, range(obs_dim - 2, obs_dim), axis=1)
        goal = tf.gather(obs, range(obs_dim - 6, obs_dim - 4), axis=1)
        obstacles = tf.gather(obs, range(obs_dim - 6 - 3 * num_obstacles, obs_dim - 6), axis=1)
        return {'x': x, 'xd': xd, 'goal': goal, 'obstacles': obstacles}

    def obs_to_value_input(self, obs):
        obs = tf.cast(obs, self.dtype)
        batch_size, obs_dim = obs.shape
        batch_size, obs_dim = int(batch_size), int(obs_dim)
        obs_to_value = tf.gather(obs, range(0, obs_dim - 6), axis=1)
        return obs_to_value

    @staticmethod
    def policy_output_to_action(actions):
        return actions


class ThreeLinkFullRMPWrapper(EnvWrapper):
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def obs_to_policy_input(self, obs):
        obs = tf.cast(obs, self.dtype)
        batch_size, obs_dim = obs.shape
        batch_size, obs_dim = int(batch_size), int(obs_dim)

        num_obstacles = int((obs_dim - 11) / 5)

        sin_q = tf.gather(obs, range(0, 3), axis=1)
        cos_q = tf.gather(obs, range(3, 6), axis=1)
        q = tf.atan2(sin_q, cos_q)
        qd = tf.gather(obs, range(6, 9), axis=1)
        obstacles = tf.gather(obs, range(obs_dim - 3 * num_obstacles, obs_dim), axis=1)
        return {'q': q, 'qd': qd, 'goal': None, 'obstacles': obstacles}

    def obs_to_value_input(self, obs):
        obs = tf.cast(obs, self.dtype)
        batch_size, obs_dim = obs.shape
        batch_size, obs_dim = int(batch_size), int(obs_dim)
        obs_to_value = tf.gather(obs, range(0, obs_dim - 6), axis=1)
        return obs_to_value

    @staticmethod
    def policy_output_to_action(actions):
        return actions


class FrankaWrapper(EnvWrapper):
    def __init__(self, dtype=tf.float32):
        self.dtype = dtype

    def obs_to_policy_input(self, obs):
        return tf.cast(obs, self.dtype)

    def obs_to_value_input(self, obs):
        return tf.cast(obs, self.dtype)

    @staticmethod
    def policy_output_to_action(actions):
        return actions


class FrankaRMPWrapper(EnvWrapper):
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def obs_to_policy_input(self, obs):
        obs = tf.cast(obs, self.dtype)
        batch_size, obs_dim = obs.shape
        batch_size, obs_dim = int(batch_size), int(obs_dim)

        num_obstacles = int((obs_dim - 33) / 7)

        x = tf.gather(obs, range(obs_dim - 6, obs_dim - 3), axis=1)
        xd = tf.gather(obs, range(obs_dim - 3, obs_dim), axis=1)
        goal = tf.gather(obs, range(obs_dim - 9, obs_dim - 6), axis=1)
        obstacles = tf.gather(obs, range(obs_dim - 9 - 4 * num_obstacles, obs_dim - 9), axis=1)
        return {'x': x, 'xd': xd, 'goal': goal, 'obstacles': obstacles}

    def obs_to_value_input(self, obs):
        obs = tf.cast(obs, self.dtype)
        batch_size, obs_dim = obs.shape
        batch_size, obs_dim = int(batch_size), int(obs_dim)
        obs_to_value = tf.gather(obs, range(0, obs_dim - 9), axis=1)
        return obs_to_value

    @staticmethod
    def policy_output_to_action(actions):
        return actions

class FrankaFullRMPWrapper(EnvWrapper):
    def __init__(self, dtype='float32'):
        self.dtype = dtype

    def obs_to_policy_input(self, obs):
        obs = tf.cast(obs, self.dtype)
        batch_size, obs_dim = obs.shape
        batch_size, obs_dim = int(batch_size), int(obs_dim)

        num_obstacles = int((obs_dim - 24) / 7)

        sin_q = tf.gather(obs, range(0, 7), axis=1)
        cos_q = tf.gather(obs, range(7, 14), axis=1)
        q = tf.atan2(sin_q, cos_q)
        qd = tf.gather(obs, range(14, 21), axis=1)
        obstacles = tf.gather(obs, range(obs_dim - 4 * num_obstacles, obs_dim), axis=1)
        return {'q': q, 'qd': qd, 'goal': None, 'obstacles': obstacles}

    def obs_to_value_input(self, obs):
        obs = tf.cast(obs, self.dtype)
        batch_size, obs_dim = obs.shape
        batch_size, obs_dim = int(batch_size), int(obs_dim)
        obs_to_value = tf.gather(obs, range(0, obs_dim - 9), axis=1)
        return obs_to_value

    @staticmethod
    def policy_output_to_action(actions):
        return actions