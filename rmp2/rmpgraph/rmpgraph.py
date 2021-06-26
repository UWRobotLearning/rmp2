"""
Base class for rmp2 graph
"""

from rmp2.utils.tf_utils import ip, ipm, solve, gradient, batch_jacobian
from rmp2.utils.python_utils import timing
import tensorflow as tf
from abc import ABC, abstractmethod


class RMPGraph(tf.Module, ABC):
    """
    tensorflow module for RMP^2
    """
    def __init__(self, rmps, rmp_type='canonical', timed=False, offset=1e-3, dtype=tf.float32, name='rmpgraph', **kwargs):
        super(RMPGraph, self).__init__(name=name)

        self.rmps = rmps
        self.rmp_type = rmp_type
        self.timed = timed

        self.offset = offset


    @abstractmethod
    def forward_mapping(self, q, **features):
        """
        forward mapping from root node to leaf nodes given environment features
        --------------------------------------------
        :param q: root node coordinate
        :param features: environment features, lists/dicts, e.g. goals, obstacles, etc.
        :return xs: list of leaf node coordinates
        """

    def rmp_evals(self, xs, xds, **features):
        """
        evaluate the geometry at leaf nodes given environment features
        --------------------------------------------
        :param xs: list of leaf node coordinates
        :param xds: list of leaf node generalized velocities
        :param features: environment features, lists/dicts, e.g. goals, obstacles, etc.
        :return metrics: list of leaf node metrics
        :return forces/accelerations: list of leaf node forces/accelerations
        """
        metrics = []
        accelerations = []

        # loop over leaf nodes and evaulate rmp at each leaf node
        for (x, xd, rmp) in zip(xs, xds, self.rmps):
            metric, acceleration = rmp(x, xd, rmp_type=self.rmp_type, **features)
            metrics.append(metric)
            accelerations.append(acceleration)

            if not tf.math.reduce_all(tf.math.is_finite(metric)):
                tf.print('!!!nan or inf in leaf metric!!!', rmp.name)
            if not tf.math.reduce_all(tf.math.is_finite(acceleration)):
                tf.print('!!!nan or inf in leaf force!!!!', rmp.name)

        return metrics, accelerations


    def solve(self, q, qd, method='rmp2', mode='value', **features):
        if method == 'rmp2':
            return self.solve_rmp2(q, qd, mode=mode, **features)
        elif method == 'direct':
            return self.solve_direct(q, qd, mode=mode, **features)
        else:
            raise ValueError

    @tf.function
    def solve_rmp2(self, q, qd, mode='value', **features):
        tf.compat.v1.enable_v2_behavior()
        print('------------buiding graph--------------')

        with timing('<rmp2> forward pass', self.timed):
            # compute the generalized coordinates, velocities, and curvatures
            # at the leaf nodes

            with tf.GradientTape(watch_accessed_variables=False) as gg:
                gg.watch(q)
                with tf.GradientTape(watch_accessed_variables=False) as ggg:
                    ggg.watch(q)
                    # generalized coordinates through forward mapping
                    xs = self.forward_mapping(q, **features)
                    dummy_ones = [tf.ones_like(x) for x in xs]
                    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
                        g.watch(dummy_ones)
                        sum_x = ip(xs, dummy_ones)
                        sum_xd = ip(qd, [gradient(ggg, sum_x, q)])
                        grd = gradient(gg, sum_xd, q)
                        sum_crv = ip(qd, grd)

            # curvature terms
            xds = gradient(g, sum_xd, dummy_ones)
            crvs = gradient(g, sum_crv, dummy_ones)

            for crv, rmp in zip(crvs, self.rmps):
                if not tf.math.reduce_all(tf.math.is_finite(crv)):
                    tf.print('!!!nan or inf in curvature!!!', rmp.name)

        # evaluate the rmps at the leaf nodes
        with timing('<rmp2> rmp evaluation', self.timed):
            if self.rmp_type == 'canonical':
                mtr_leafs, acc_leafs = self.rmp_evals(xs, xds, **features)
            elif self.rmp_type == 'natural':
                mtr_leafs, fce_leafs = self.rmp_evals(xs, xds, **features)
            else:
                raise ValueError

        # compute pullback metric and rhs
        with timing('<rmp2> backward pass', self.timed):
            dummy_q = q + 0.

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
                g.watch(dummy_q)
                dummy_xs = self.forward_mapping(dummy_q, **features)
                with tf.GradientTape(watch_accessed_variables=False) as gg:
                    gg.watch(q)
                    xs = self.forward_mapping(q, **features)
                    y = ipm(dummy_xs, xs, mtr_leafs)
                grd = gradient(gg, y, q)

                # right hand side
                if self.rmp_type == 'canonical':
                    z = ipm(dummy_xs, acc_leafs, mtr_leafs) - ipm(dummy_xs, crvs, mtr_leafs)
                elif self.rmp_type == 'natural':
                    z = ip(dummy_xs, fce_leafs) - ipm(dummy_xs, crvs, mtr_leafs)
            if not tf.math.reduce_all(tf.math.is_finite(z)):
                tf.print('!!!nan or inf in z!!!')

            f = gradient(g, z, dummy_q)
            if not tf.math.reduce_all(tf.math.is_finite(f)):
                    tf.print('!!!nan or inf in root force (before normalization) !!!')
            # pullback metrics
            M = batch_jacobian(g, grd, dummy_q)
            if not tf.math.reduce_all(tf.math.is_finite(f)):
                    tf.print('!!!nan or inf in root metric (before normalization) !!!')
            # solve for generalized acceleration at the root

        if mode == 'tuple':
            return M, f
        elif mode == 'value':
            with timing('<rmp2> resolve', self.timed):
                # scale both the metric and the force for the same amount
                M_max = tf.reduce_max(tf.abs(M), [1, 2]) * 0.01
                M_max = tf.maximum(M_max, tf.ones_like(M_max))
                M = tf.einsum('b, bij->bij', 1. / M_max, M)
                f = tf.einsum('b, bi->bi', 1. / M_max, f)

                # add a small offset to the diagonal for numerical stability
                M = M + self.offset * tf.eye(M.shape[1], batch_shape=(M.shape[0],), dtype=M.dtype)

                if not tf.math.reduce_all(tf.math.is_finite(M)):
                    tf.print('!--------nan in root metric--------!')
                if not tf.math.reduce_all(tf.math.is_finite(f)):
                    tf.print('!--------nan in root force--------!')

                qdd = solve(M, f)

                if not tf.math.reduce_all(tf.math.is_finite(qdd)):
                    tf.print('!--------nan in root acceleration--------!')
            return qdd
        else:
            raise ValueError


    @tf.function
    def solve_direct(self, q, qd, mode='value', **features):
        with timing('<direct> forward pass', self.timed):
            with tf.GradientTape(watch_accessed_variables=False) as gg:
                gg.watch(q)
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as ggg:
                    ggg.watch(q)
                    xs = self.forward_mapping(q, **features)
                    dummy_ones = [tf.ones_like(x) for x in xs]
                # compute jacobians
                jacs = [batch_jacobian(ggg, x, q) for x in xs]
                # del ggg
                xds = [tf.linalg.matvec(jac, qd) for jac in jacs]
                with tf.GradientTape(watch_accessed_variables=False) as g:
                    g.watch(dummy_ones)
                    sum_xd = ip(dummy_ones, xds)
                    grd = gradient(gg, sum_xd, q)
                    sum_crv = ip(qd, grd)
            crvs = gradient(g, sum_crv, dummy_ones)

        # evaluate the rmps at the leaf nodes
        with timing('<direct> rmp evaluation', self.timed):
            if self.rmp_type == 'canonical':
                mtr_leafs, acc_leafs = self.rmp_evals(xs, xds, **features)
            elif self.rmp_type == 'natural':
                mtr_leafs, fce_leafs = self.rmp_evals(xs, xds, **features)
            else:
                raise ValueError

        # compute pullback metric and forces
        with timing('<direct> pullback', self.timed):
            M = sum(tf.einsum('bji, bjk, bkl->bil', jac, mtr_leaf, jac) for (jac, mtr_leaf) in zip(jacs, mtr_leafs))

            if self.rmp_type == 'canonical':
                f = sum(
                    tf.einsum(
                        'bji, bjk, bk->bi',
                        jac, mtr_leaf, acc_leaf - crv
                        ) for (jac, mtr_leaf, acc_leaf, crv) in zip(jacs, mtr_leafs, acc_leafs, crvs))
            elif self.rmp_type == 'natural':
                f = sum(
                    tf.einsum(
                        'bji, bj->bi',
                        jac, fce_leaf) for (jac, fce_leaf) in zip(jacs, fce_leafs)) - \
                    sum(
                        tf.einsum(
                            'bji, bjk, bk->bi',
                            jac, mtr_leaf, crv) for (jac, mtr_leaf, crv) in zip(jacs, mtr_leafs, crvs))
            else:
                raise ValueError

        if mode == 'tuple':
            return M, f
        elif mode == 'value':
            with timing('<direct> resolve', self.timed):
                # scale both the metric and the force for the same amount
                M_max = tf.reduce_max(tf.abs(M), [1, 2]) * 0.01
                M = tf.einsum('b, bij->bij', 1. / M_max, M)
                f = tf.einsum('b, bi->bi', 1. / M_max, f)

                # add a small offset to the diagonal for numerical stability
                M = M + self.offset * tf.eye(M.shape[1], batch_shape=(M.shape[0],), dtype=M.dtype)

                if not tf.math.reduce_all(tf.math.is_finite(M)):
                    tf.print('!--------nan in root metric--------!')
                if not tf.math.reduce_all(tf.math.is_finite(f)):
                    tf.print('!--------nan in root force--------!')

                qdd = solve(M, f)

                if not tf.math.reduce_all(tf.math.is_finite(qdd)):
                    tf.print('!--------nan in root acceleration--------!')
                return qdd
        else:
            raise ValueError

    def __call__(self, q, qd, **features):
        return self.solve(q, qd, **features)
