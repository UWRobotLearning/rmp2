"""
helper functions for computing the distance between balls 
and between balls and boxes
"""

from rmp2.utils.tf_utils import pdist2
import tensorflow as tf


def distmap(type, control_points, control_point_radii, **params):
    if type == 'ball':
        return dist2balls(control_points, control_point_radii, **params)
    elif type == 'box':
        return dist2boxes(control_points, control_point_radii, **params)
    else:
        raise ValueError

def dist2balls(control_points, control_point_radii, obstacle_centers, obstacle_radii):
    """
    compute the distance between control point balls and obstacle balls
    :param control_points: batch_size x num_control_points x dimension, control point positions
    :param control_point_radii: 1 x num_control_points, control point ball radii
    obstacle_centers: batch_size x num_obstacles x dimension, obstacle center positions
    obstacle_radii: batch_size x num_obstacles, obstacle radii
    :return obstacle_dists: batch_size x num_control_points x num_obstacles, 
    distance between control point balls and obstacle boxes
    """
    center_dists = pdist2(control_points, obstacle_centers)
    obstacle_radii = tf.expand_dims(obstacle_radii, 1)
    arm_radii = tf.expand_dims(control_point_radii, -1)
    collision_radii = arm_radii + obstacle_radii

    obstacle_dists = center_dists - collision_radii
    obstacle_dists = tf.reshape(obstacle_dists, (-1, 1))
    
    return obstacle_dists

def dist2boxes(control_points, control_point_radii, obstacle_mins, obstacle_maxs):
    """
    compute the distance between control point balls and obstacle boxes
    :param control_points: batch_size x num_control_points x dimension, control point positions
    :param control_point_radii: 1 x num_control_points, control point ball radii
    :param obstacle_mins: batch_size x num_obstacles x dimension, lower corners of obstacle boxes
    :param obstacle_maxs: batch_size x num_obstacles x dimension, upper corner of obstacle boxes
    :return obstacle_dists: batch_size x num_control_points x num_obstacles, 
    distance between control point balls and obstacle boxes
    """
    control_points = tf.expand_dims(control_points, 2)
    obstacle_mins = tf.expand_dims(obstacle_mins, 1)
    obstacle_maxs = tf.expand_dims(obstacle_maxs, 1)
    control_points_proj = tf.minimum(
        tf.maximum(control_points, obstacle_mins), 
        obstacle_maxs)
    obstacle_dists = tf.norm(control_points - control_points_proj, axis=-1)
    obstacle_dists = obstacle_dists - tf.expand_dims(tf.expand_dims(control_point_radii, -1), -1)
    obstacle_dists = tf.reshape(obstacle_dists, (-1, 1))

    return obstacle_dists