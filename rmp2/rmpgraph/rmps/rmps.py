"""
hand-designed and learnable rmps
"""

import tensorflow as tf
from abc import abstractmethod

def get_rmp(rmp_type, params, dtype):
    """
    return an rmp instance given type and params
    :param rmp_type: str, rmp type
    :param params: dict, rmp parameters
    :param dtype: str, e.g. float32 or float64
    """
    if rmp_type == 'cspace_target_rmp':
        rmp = CSpaceTarget(dtype=dtype, **params)
    elif rmp_type == 'joint_limit_rmp':
        rmp = JointLimit(dtype=dtype, **params)
    elif rmp_type == 'joint_velocity_cap_rmp':
        rmp = JointVelocityCap(dtype=dtype, **params)
    elif rmp_type == 'target_rmp':
        rmp = TargetAttractor(dtype=dtype, **params)
    elif rmp_type == 'collision_rmp':
        rmp = ObstacleAvoidance(dtype=dtype, **params)
    elif rmp_type == 'damping_rmp':
        rmp = JointDamping(dtype=dtype, **params)
    else:
        raise ValueError
    return rmp

class RMP(tf.Module):
    """
    base rmp class
    """
    def __init__(self, feature_keys=None, name='rmp', dtype=tf.float32):
        super().__init__(name=name)
        """
        :param feature_keys: keys for features as a list, e.g. ['goal', 'obstacles']
        :param name: name for the module
        :param dtype: dtype, str or tf.floatX
        """
        if feature_keys is None:
            self.feature_keys = []
        else:
            self.feature_keys = feature_keys
        self.dtype = dtype


    def rmp_eval(self, x, xd, rmp_type='canonical', **features):
        """
        evaluating the rmp node
        :param x: tf.Tensor, position
        :param xd: tf.Tensor, velocity
        :param rmp_type: str, canonical or natural
        :param features: dict, all features as a dictionary
        """
        # only select the features specified by feature_keys
        selected_features = {key: features[key] for key in self.feature_keys}
        if rmp_type == 'canonical':
            return self.rmp_eval_canonical(x, xd, **selected_features)
        elif rmp_type == 'natural':
            return self.rmp_eval_natural(x, xd, **selected_features)

    @abstractmethod
    def rmp_eval_canonical(self, x, xd, **features):
        """
        canonical rmp eval
        :param x: tf.Tensor position
        :param xd: tf.Tensor velocity
        :param features: dict, all selected features
        :return metric: rmp importance weight
        :return acceleration: rmp acceleration
        """

    @abstractmethod
    def rmp_eval_natural(self, x, xd, **features):
        """
        natural rmp eval
        :param x: tf.Tensor position
        :param xd: tf.Tensor velocity
        :param features: dict, all selected features
        :return metric: rmp importance weight
        :return force: rmp force (importance weight * acceleration)
        """

    def __call__(self, x, xd, rmp_type='canonical', **features):
        return self.rmp_eval(x, xd, rmp_type, **features)



class CSpaceTarget(RMP):
    """
    configuration space target reaching (default configuration)
    """
    def __init__(self, metric_scalar, position_gain, damping_gain,
        robust_position_term_thresh, inertia,
        name='cspace_target', dtype=tf.float32):

        super(CSpaceTarget, self).__init__(name=name, dtype=dtype)
        self.metric_scalar = metric_scalar
        self.position_gain = position_gain
        self.damping_gain = damping_gain
        self.robust_position_term_thresh = robust_position_term_thresh
        self.inertia = inertia

    def rmp_eval_canonical(self, x, xd, **features):
        batch_size, x_shape = x.shape

        x_hat, x_norm = tf.linalg.normalize(x, axis=1)
        qdd_position = tf.where(
            x_norm < self.robust_position_term_thresh,
            -x * self.position_gain,
            -self.robust_position_term_thresh * x_hat * self.position_gain,
            )
        qdd_velocity = -self.damping_gain * xd
        eye = tf.eye(x_shape, batch_shape=[batch_size], dtype=self.dtype)
        metric = eye * (self.metric_scalar + self.inertia)
        acceleration = qdd_position + qdd_velocity
        return metric, acceleration

    def rmp_eval_natural(self, x, xd, **features):
        metric, acceleration = self.rmp_eval_canonical(x, xd, **features)
        force = (self.metric_scalar + self.inertia) * acceleration
        return metric, force


class JointLimit(RMP):
    def __init__(
        self,
        metric_scalar, metric_length_scale,
        metric_exploder_eps, metric_velocity_gate_length_scale,
        accel_damper_gain, accel_potential_gain,
        accel_potential_exploder_eps, accel_potential_exploder_length_scale,
        name='joint_limit', dtype=tf.float32):

        super(JointLimit, self).__init__(name=name, dtype=dtype)
        self.metric_scalar = metric_scalar
        self.metric_length_scale = metric_length_scale
        self.metric_exploder_eps = metric_exploder_eps
        self.metric_velocity_gate_length_scale = metric_velocity_gate_length_scale

        self.accel_damper_gain = accel_damper_gain
        self.accel_potential_gain = accel_potential_gain
        self.accel_potential_exploder_eps = accel_potential_exploder_eps
        self.accel_potential_exploder_length_scale = accel_potential_exploder_length_scale

    def rmp_eval_canonical(self, x, xd, **features):
        x = tf.maximum(x, tf.zeros_like(x))
        metric_before_gate = self.metric_scalar / (x / self.metric_length_scale + self.metric_exploder_eps)
        metric = (1. - tf.sigmoid(xd / self.metric_velocity_gate_length_scale)) * metric_before_gate
        metric = tf.expand_dims(metric, -1)
        xdd_velocity = -self.accel_damper_gain * xd
        scaled_x = x / self.accel_potential_exploder_length_scale
        xdd_position = self.accel_potential_gain / (scaled_x * scaled_x + self.accel_potential_exploder_eps)
        acceleration = xdd_position + xdd_velocity

        return metric, acceleration

    def rmp_eval_natural(self, x, xd, **features):
        metric, acceleration = self.rmp_eval_canonical(x, xd, **features)
        force = tf.squeeze(metric, -1) * acceleration
        return metric, force


class JointVelocityCap(RMP):
    def __init__(
        self, max_velocity, velocity_damping_region, damping_gain, metric_weight,
        eps=1e-6,
        name='joint_velocity_cap', dtype=tf.float32):

        super(JointVelocityCap, self).__init__(name=name, dtype=dtype)
        self.max_velocity = max_velocity
        self.velocity_damping_region = velocity_damping_region
        self.damping_gain = damping_gain
        self.metric_weight = metric_weight
        self.eps = eps
        self.damped_velocity_cutoff = self.max_velocity - self.velocity_damping_region


    def rmp_eval_canonical(self, x, xd, **features):
        batch_size, x_shape = x.shape

        delta_velocity = tf.abs(xd) - self.damped_velocity_cutoff
        xdd = - tf.abs(self.damping_gain * delta_velocity) * tf.sign(xd)

        clipped_relative_velocity = tf.minimum(delta_velocity, self.velocity_damping_region - self.eps)

        velocity_ratio = clipped_relative_velocity / self.velocity_damping_region
        metric = self.metric_weight / (1.0 - (velocity_ratio ** 2))

        metric = tf.where(tf.abs(xd) < self.damped_velocity_cutoff, tf.zeros_like(metric), metric)
        metric = tf.expand_dims(metric, -1)
        acceleration = tf.where(tf.abs(xd) < self.damped_velocity_cutoff, tf.zeros_like(xdd), xdd)
        return metric, acceleration

    def rmp_eval_natural(self, x, xd, **features):
        metric, acceleration = self.rmp_eval_canonical(x, xd, **features)
        force = tf.squeeze(metric, -1) * acceleration
        return metric, force


class TargetAttractor(RMP):
    def __init__(
        self,
        accel_p_gain, accel_d_gain,
        accel_norm_eps, metric_alpha_length_scale,
        min_metric_alpha, max_metric_scalar, min_metric_scalar,
        proximity_metric_boost_scalar,  proximity_metric_boost_length_scale,
        name='attractor', dtype=tf.float32):
        super(TargetAttractor, self).__init__(feature_keys=['goal'], name=name, dtype=dtype)

        self.accel_p_gain = accel_p_gain
        self.accel_d_gain = accel_d_gain
        self.accel_norm_eps = accel_norm_eps
        self.metric_alpha_length_scale = metric_alpha_length_scale
        self.min_metric_alpha = min_metric_alpha
        self.max_metric_scalar = max_metric_scalar
        self.min_metric_scalar = min_metric_scalar
        self.proximity_metric_boost_scalar = proximity_metric_boost_scalar
        self.proximity_metric_boost_length_scale = proximity_metric_boost_length_scale

    def rmp_eval_canonical(self, x, xd, **features):
        batch_size, n_dims = x.shape
        batch_size, n_dims = int(batch_size), int(n_dims)

        delta = features['goal'] - x
        delta_norm = tf.linalg.norm(delta, axis=1)
        delta_norm = tf.expand_dims(delta_norm, -1)
        soft_delta_norm = tf.maximum(delta_norm, self.accel_norm_eps / 10 * tf.ones_like(delta_norm))
        delta_hat = delta / soft_delta_norm

        accel = self.accel_p_gain * delta / (delta_norm + self.accel_norm_eps) - self.accel_d_gain * xd

        eye = tf.eye(n_dims, batch_shape=[batch_size], dtype=self.dtype)
        S = tf.einsum('bi, bj->bij', delta_hat, delta_hat)
        scaled_dist = delta_norm / self.metric_alpha_length_scale
        a = (1. - self.min_metric_alpha) * tf.exp(-.5 * scaled_dist * scaled_dist) + self.min_metric_alpha
        a = tf.expand_dims(a, -1)
        metric = a * self.max_metric_scalar * eye + (1. - a) * self.min_metric_scalar * S

        boost_scaled_dist = delta_norm / self.proximity_metric_boost_length_scale
        boost_a = tf.exp(-.5 * boost_scaled_dist * boost_scaled_dist)
        metric_boost_scalar = boost_a * self.proximity_metric_boost_scalar + (1. - boost_a) * 1.
        metric_boost_scalar = tf.expand_dims(metric_boost_scalar, -1)
        metric = metric_boost_scalar * metric
        return metric, accel

    def rmp_eval_natural(self, x, xd, **features):
        metric, acceleration = self.rmp_eval_canonical(x, xd, **features)
        force = tf.einsum('bij, bj->bi', metric, acceleration)
        return metric, force


class ObstacleAvoidance(RMP):
    def __init__(
        self,
        margin,
        damping_gain,
        damping_std_dev,
        damping_robustness_eps,
        damping_velocity_gate_length_scale,
        repulsion_gain,
        repulsion_std_dev,
        metric_modulation_radius,
        metric_scalar,
        metric_exploder_std_dev,
        metric_exploder_eps,
        name='obstacle_avoidance', dtype=tf.float32):

        super(ObstacleAvoidance, self).__init__(name=name, dtype=dtype)
        self.margin = margin
        self.damping_gain = damping_gain
        self.damping_std_dev = damping_std_dev
        self.damping_robustness_eps = damping_robustness_eps
        self.damping_velocity_gate_length_scale = damping_velocity_gate_length_scale
        self.repulsion_gain = repulsion_gain
        self.repulsion_std_dev = repulsion_std_dev
        self.metric_modulation_radius = metric_modulation_radius
        self.metric_scalar = metric_scalar
        self.metric_exploder_std_dev = metric_exploder_std_dev
        self.metric_exploder_eps = metric_exploder_eps

    def _smooth_activation_gate(self, x):
        r = self.metric_modulation_radius
        gate = x * x / (r * r) - 2. * x / r + 1.
        gate = tf.where(x > r, tf.zeros_like(gate), gate)
        return gate

    def _length_scale_normalized_repulsion_distance(self, x):
        return x / self.repulsion_std_dev

    def _calc_damping_gain_divisor(self, x):
        z = x / self.damping_std_dev + self.damping_robustness_eps
        return z

    def rmp_eval_canonical(self, x, xd, **features):
        x = x - self.margin
        x = tf.maximum(x, tf.zeros_like(x))
        base_metric = self.metric_scalar / (x / self.metric_exploder_std_dev + self.metric_exploder_eps)
        metric = base_metric * self._smooth_activation_gate(x)
        xdd_repel = self.repulsion_gain * tf.exp(-self._length_scale_normalized_repulsion_distance(x))
        sig = tf.sigmoid(xd / self.damping_velocity_gate_length_scale)
        xdd_damping = -(1. - sig) * self.damping_gain * xd / self._calc_damping_gain_divisor(x)

        accel = xdd_repel + xdd_damping
        metric = tf.where(x > self.metric_modulation_radius, tf.zeros_like(metric), (1 - sig) * metric)
        metric = tf.expand_dims(metric, -1)
        return metric, accel

    def rmp_eval_natural(self, x, xd, **features):
        metric, acceleration = self.rmp_eval_canonical(x, xd, **features)
        force = tf.squeeze(metric, -1) * acceleration
        return metric, force


class JointDamping(RMP):
    def __init__(
        self,
        accel_d_gain, metric_scalar, inertia,
        name='joint_damping', dtype=tf.float32):

        super(JointDamping, self).__init__(name=name, dtype=dtype)
        self.accel_d_gain = accel_d_gain
        self.metric_scalar = metric_scalar
        self.inertia = inertia

    def rmp_eval_canonical(self, x, xd, **features):
        batch_size, x_shape = x.shape

        xd_norm = tf.norm(xd, axis=1, keepdims=True)
        nonlinear_gain = self.accel_d_gain * xd_norm
        acceleration = -nonlinear_gain * xd
        nonlinear_metric_scalar = self.metric_scalar * xd_norm
        nonlinear_metric_scalar = tf.expand_dims(nonlinear_metric_scalar, -1)
        metric = tf.eye(x_shape, batch_shape=[batch_size], dtype=self.dtype) * (nonlinear_metric_scalar + self.inertia)

        return metric, acceleration

    def rmp_eval_natural(self, x, xd, **features):
        batch_size, x_shape = x.shape

        xd_norm = tf.norm(xd, axis=1, keepdims=True)
        nonlinear_gain = self.accel_d_gain * xd_norm
        acceleration = -nonlinear_gain * xd
        nonlinear_metric_scalar = self.metric_scalar * xd_norm
        nonlinear_metric_scalar = tf.expand_dims(nonlinear_metric_scalar, -1)
        metric = tf.eye(x_shape, batch_shape=[batch_size], dtype=self.dtype) * (nonlinear_metric_scalar + self.inertia)

        force = acceleration * (nonlinear_metric_scalar + self.inertia)

        return metric, force


class ExternalCanonicalRMP(RMP):
    def __init__(
        self,
        name,
        link, 
        handcrafted_rmp = None,
        handcrafted_rmp_config = None, 
        identity_multiplier = 0.,
        dtype=tf.float32):
        super().__init__(feature_keys=[name], name=name, dtype=dtype)
        self.link = link
        self.feature_name = name
        if handcrafted_rmp is not None:
            self.handcrafted_rmp = get_rmp(handcrafted_rmp, handcrafted_rmp_config, dtype)
            self.feature_keys += self.handcrafted_rmp.feature_keys
        else:
            self.handcrafted_rmp = None
            self.identity_multiplier = identity_multiplier

    def rmp_eval_canonical(self, x, xd, **features):
        metric_sqrt, acceleration = features[self.feature_name]
        if self.handcrafted_rmp is not None:
            handcrafted_metric, handcrafted_acceleration = self.handcrafted_rmp(x, xd, **features)
            handcrafted_metric_sqrt = tf.linalg.cholesky(handcrafted_metric)
            metric_sqrt = metric_sqrt + handcrafted_metric_sqrt
            acceleration = acceleration + handcrafted_acceleration
        metric = tf.einsum('bij, bkj -> bik', metric_sqrt, metric_sqrt)
        if self.handcrafted_rmp is None:
            batch_size, x_shape = x.shape
            batch_size, x_shape = int(batch_size), int(x_shape)
            metric = metric + self.identity_multiplier * tf.eye(x_shape, batch_shape=[batch_size], dtype=self.dtype)
        return metric, acceleration

    def rmp_eval_natural(self, x, xd, **features):
        metric, acceleration = self.rmp_eval_canonical(x, xd, **features)
        force = tf.einsum('bij,bj->bi', metric, acceleration)
        return metric, force