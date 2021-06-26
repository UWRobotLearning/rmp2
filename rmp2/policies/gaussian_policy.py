# import tensorflow as tf

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from rmp2.policies.policy_networks import get_policy_network
from rmp2.utils.env_wrappers import load_env_wrapper
import numpy as np

tf1, tf, tfv = try_import_tf()

class DiagGaussianModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 policy_config, value_config, env_wrapper, init_lstd=-1., min_std=1e-12, clip_mean=None, dtype=tf.float32):
        """
        not use num_outputs
        """
        super().__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.num_outputs = num_outputs

        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]

        self.bounded = np.logical_and(action_space.bounded_above,
                                      action_space.bounded_below).any()

        self.clip_mean = clip_mean
        if self.clip_mean == "squash":
            self.action_range = tf.constant((action_space.high - action_space.low)[None])
            self.low_action = tf.constant(action_space.low[None])
        elif self.clip_mean == "clip":
            self.action_space_high = tf.constant(action_space.high[None])
            self.action_space_low = tf.constant(action_space.low[None])

        if type(env_wrapper) == str:
            env_wrapper = load_env_wrapper(env_wrapper, dtype=dtype)
        self.env_wrapper = env_wrapper

        
        _dummy_obs = tf.zeros((1, obs_dim), dtype=dtype)
        _dummy_policy_input = self.env_wrapper.obs_to_policy_input(_dummy_obs)
        _dummy_value_input = self.env_wrapper.obs_to_value_input(_dummy_obs)
        if type(_dummy_policy_input) == dict:
            policy_input_dim = None
        else:
            policy_input_dim = int(_dummy_policy_input.shape[1])
        value_input_dim = int(_dummy_value_input.shape[1])

        self.policy_net = get_policy_network(policy_input_dim, action_dim, policy_config, name=name+'_policy', dtype=dtype) # todo: initialize policy network
        self.value_net = get_policy_network(value_input_dim, 1, value_config, name=name+'_value', dtype=dtype) # todo: initialize value network

        init_lstd = np.broadcast_to(init_lstd, action_dim)
        self._lstd = tf.Variable(init_lstd, dtype=dtype)

        self._min_lstd = tf.constant(np.log(min_std), dtype=dtype)

        variables = self.policy_net.variables + self.value_net.variables + (self._lstd,)
        self.register_variables(variables)

    def _policy(self, input_dict):
        policy_input = self.env_wrapper.obs_to_policy_input(input_dict["obs"])
        policy_output = self.policy_net(policy_input)
        action = self.env_wrapper.policy_output_to_action(policy_output)
        return action

    def _value(self, input_dict):
        value_input = self.env_wrapper.obs_to_value_input(input_dict["obs"])
        value = self.value_net(value_input)
        return value

    def forward(self, input_dict, state, seq_lens):
        x = self._policy(input_dict)
        if self.bounded and self.clip_mean == "squash":
            sigmoid_out = tf.nn.sigmoid(2 * x)
            mean = self.action_range * sigmoid_out + self.low_action
        if self.clip_mean == "clip":
            mean = tf.clip_by_value(x, self.action_space_low, self.action_space_high)
        else:
            mean = x
        lstd = tf.zeros_like(mean) + self.lstd
        model_out = tf.concat([mean, lstd], axis=1)
        
        self._value_out = self._value(input_dict)
        assert model_out.shape[1] == self.num_outputs
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @property
    def lstd(self):
        return tf.maximum(self._lstd, self._min_lstd)