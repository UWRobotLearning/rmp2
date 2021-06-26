from rmp2.utils.tf_utils import MLP
import tensorflow as tf
import numpy as np

def get_policy_network(x_shape, y_shape, config, dtype, name='policy_network'):
    if config['model'] == 'mlp':
        del config['model']
        network = MLP(x_shape, y_shape, name=name, dtype=dtype, **config)
    elif config['model'] == 'flat_rmp':
        del config['model']
        network = FlatRMPPolicy(
            x_shape, y_shape, 
            rmp_config=config,
            dtype=dtype, name=name)
    else:
        raise ValueError
    return network


class FlatRMPCanonical(tf.Module):
    def __init__(self, x_shape, feature_shape=0, feature_keys=None, units=(), activation='tanh', 
        max_metric_value=10., max_accel_value=3.,
        hidden_layer_init_scale=2.0, output_layer_init_scale=0.1, init_distribution='uniform',
        dtype=tf.float32, name='flat_rmp'):

        super().__init__(name=name)

        if feature_keys is None:
            self.feature_keys = []
        else:
            self.feature_keys = feature_keys
        self.dtype = dtype

        self.net = MLP(
            x_shape * 2 + feature_shape, x_shape ** 2 + x_shape, units, activation=activation,
            hidden_layer_init_scale=hidden_layer_init_scale, output_layer_init_scale=output_layer_init_scale, init_distribution='uniform',
            dtype=dtype, name=name+'_mlp')
        
        rmp_limit = np.array([[np.sqrt(max_metric_value)] * x_shape ** 2 + [max_accel_value] * x_shape])
        self.rmp_limit = tf.convert_to_tensor(rmp_limit, dtype=dtype)

    def __call__(self, x, xd, **features):
        features = {key: features[key] for key in self.feature_keys}

        state = tf.concat((x, xd) + tuple(features.values()), axis=1)
        flat_rmp = self.net(state)
        flat_rmp = tf.nn.tanh(flat_rmp) * self.rmp_limit
        return flat_rmp


class FlatRMPPolicy(tf.Module):
    def __init__(self, x_shape, y_shape, rmp_config, dtype='float32', name='flat_rmpnet', **kwargs):
        super().__init__(name=name)

        self.x_shape = x_shape
        self.y_shape = y_shape

        self.rmp = FlatRMPCanonical(dtype=dtype, **rmp_config)

    def __call__(self, x):
        actions = self.rmp(**x)
        return actions