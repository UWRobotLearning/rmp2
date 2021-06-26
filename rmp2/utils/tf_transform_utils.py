# The first version was licensed as "Original Source License"(see below).
# Several enhancements and at UW Robot Learning Lab
# 
# Original Source License:
# 
# Copyright (c) 2018 mahaarbo
# Licensed under the MIT License.

"""
rigid body transformation in tensorflow
"""

import tensorflow as tf

def rotation_rpy(rpy):
    """Returns a rotation matrix from roll pitch yaw. ZYX convention."""
    cr = tf.cos(rpy[0])
    sr = tf.sin(rpy[0])
    cp = tf.cos(rpy[1])
    sp = tf.sin(rpy[1])
    cy = tf.cos(rpy[2])
    sy = tf.sin(rpy[2])

    return tf.convert_to_tensor([[cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
                                 [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
                                 [  -sp,             cp*sr,             cp*cr]])


def T_rpy(displacement, rpy):
    """Homogeneous transformation matrix with roll pitch yaw."""
    t12 = tf.concat([rotation_rpy(rpy), tf.expand_dims(displacement, -1)], axis=1)
    t3 = tf.constant([0., 0., 0., 1.], dtype=displacement.dtype)
    T = tf.concat([t12, tf.expand_dims(t3, 0)], axis=0)
    return T


def T_prismatic(xyz, rpy, axis, qi):
    # Origin rotation from RPY ZYX convention
    cr = tf.cos(rpy[0])
    sr = tf.sin(rpy[0])
    cp = tf.cos(rpy[1])
    sp = tf.sin(rpy[1])
    cy = tf.cos(rpy[2])
    sy = tf.sin(rpy[2])
    r00 = cy*cp
    r01 = cy*sp*sr - sy*cr
    r02 = cy*sp*cr + sy*sr
    r10 = sy*cp
    r11 = sy*sp*sr + cy*cr
    r12 = sy*sp*cr - cy*sr
    r20 = -sp
    r21 = cp*sr
    r22 = cp*cr
    p0 = r00*axis[0]*qi + r01*axis[1]*qi + r02*axis[2]*qi
    p1 = r10*axis[0]*qi + r11*axis[1]*qi + r12*axis[2]*qi
    p2 = r20*axis[0]*qi + r21*axis[1]*qi + r22*axis[2]*qi

    # Homogeneous transformation matrix
    t00 = tf.ones_like(qi) * r00
    t01 = tf.ones_like(qi) * r01
    t02 = tf.ones_like(qi) * r02
    t03 = xyz[0] + p0

    t10 = tf.ones_like(qi) * r10
    t11 = tf.ones_like(qi) * r11
    t12 = tf.ones_like(qi) * r12
    t13 = xyz[1] + p1

    t20 = tf.ones_like(qi) * r20
    t21 = tf.ones_like(qi) * r21
    t22 = tf.ones_like(qi) * r22
    t23 = xyz[2] + p2

    t30 = tf.zeros_like(qi)
    t31 = tf.zeros_like(qi)
    t32 = tf.zeros_like(qi)
    t33 = tf.ones_like(qi)

    T = tf.stack([
            t00, t01, t02, t03,
            t10, t11, t12, t13,
            t20, t21, t22, t23,
            t30, t31, t32, t33], axis=1)
    T = tf.reshape(T, (-1, 4, 4))
    return T


def T_revolute(xyz, rpy, axis, qi):
    # Origin rotation from RPY ZYX convention
    cr = tf.cos(rpy[0])
    sr = tf.sin(rpy[0])
    cp = tf.cos(rpy[1])
    sp = tf.sin(rpy[1])
    cy = tf.cos(rpy[2])
    sy = tf.sin(rpy[2])
    r00 = cy*cp
    r01 = cy*sp*sr - sy*cr
    r02 = cy*sp*cr + sy*sr
    r10 = sy*cp
    r11 = sy*sp*sr + cy*cr
    r12 = sy*sp*cr - cy*sr
    r20 = -sp
    r21 = cp*sr
    r22 = cp*cr

    # joint rotation from skew sym axis angle
    cqi = tf.cos(qi)
    sqi = tf.sin(qi)
    s00 = (1 - cqi)*axis[0]*axis[0] + cqi
    s11 = (1 - cqi)*axis[1]*axis[1] + cqi
    s22 = (1 - cqi)*axis[2]*axis[2] + cqi
    s01 = (1 - cqi)*axis[0]*axis[1] - axis[2]*sqi
    s10 = (1 - cqi)*axis[0]*axis[1] + axis[2]*sqi
    s12 = (1 - cqi)*axis[1]*axis[2] - axis[0]*sqi
    s21 = (1 - cqi)*axis[1]*axis[2] + axis[0]*sqi
    s20 = (1 - cqi)*axis[0]*axis[2] - axis[1]*sqi
    s02 = (1 - cqi)*axis[0]*axis[2] + axis[1]*sqi

    # Homogeneous transformation matrix
    t00 = r00*s00 + r01*s10 + r02*s20
    t10 = r10*s00 + r11*s10 + r12*s20
    t20 = r20*s00 + r21*s10 + r22*s20
    t30 = tf.zeros_like(qi)

    t01 = r00*s01 + r01*s11 + r02*s21
    t11 = r10*s01 + r11*s11 + r12*s21
    t21 = r20*s01 + r21*s11 + r22*s21
    t31 = tf.zeros_like(qi)

    t02 = r00*s02 + r01*s12 + r02*s22
    t12 = r10*s02 + r11*s12 + r12*s22
    t22 = r20*s02 + r21*s12 + r22*s22
    t32 = tf.zeros_like(qi)

    t03 = tf.ones_like(qi) * xyz[0]
    t13 = tf.ones_like(qi) * xyz[1]
    t23 = tf.ones_like(qi) * xyz[2]
    t33 = tf.ones_like(qi)

    T = tf.stack([
            t00, t01, t02, t03,
            t10, t11, t12, t13,
            t20, t21, t22, t23,
            t30, t31, t32, t33
            ], axis=1)
    T = tf.reshape(T, (-1, 4, 4))

    return T