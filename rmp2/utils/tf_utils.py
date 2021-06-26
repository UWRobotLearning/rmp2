import tensorflow as tf
import numpy as np
from collections.abc import Iterable, Sized

def ip(x, y, reduce='sum', axis=None):
    """
    inner product between vectors or list(s) of tensors
    """
    assert reduce == 'sum' or reduce is None

    # if both inputs are tensors, return inner product
    if isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
        assert x.shape == y.shape
        return tf.reduce_sum(x * y, axis=axis)

    # if one is tensor, the other is a list of tensors, return the sum (reduction=sum)
    # or a list (reduction=None) of inner products
    elif isinstance(x, tf.Tensor) and not isinstance(y, tf.Tensor):
        assert isinstance(y, Iterable)
        if reduce == 'sum':
            return sum(tf.reduce_sum(x * yy, axis=axis) for yy in y)
        elif reduce is None:
            return [tf.reduce_sum(x * yy, axis=axis) for yy in y]
    elif not isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor):
        assert isinstance(x, Iterable)
        if reduce == 'sum':
            return sum(tf.reduce_sum(xx * y, axis=axis) for xx in x)
        elif reduce is None:
            return [tf.reduce_sum(xx * y, axis=axis) for xx in x]

    # if both are lists of tensors, return the sum (reduction=sum)
    # or a list (reduction=None) of inner products
    else:
        assert isinstance(x, Iterable) and isinstance(y, Iterable)
        if isinstance(x, Sized) and isinstance(y, Sized):
            assert len(x) == len(y)
        if reduce == 'sum':
            return sum(tf.reduce_sum(xx * yy, axis=axis) for (xx, yy) in zip(x, y))
        elif reduce is None:
            return [tf.reduce_sum(xx * yy, axis=axis) for (xx, yy) in zip(x, y)]


def ipm(x, y, m, reduce='sum'):
    """
    vector-matrix-vector product x^T M y for tensors or lists of tensors
    """
    assert reduce == 'sum' or reduce is None

    # if all inputs are tensors, return vector-matrix-vector product
    if isinstance(x, tf.Tensor) and isinstance(y, tf.Tensor) and isinstance(m, tf.Tensor):
        return tf.einsum('bi,bij,bj->b', x, m, y)
    # if all inputs are lists of tensors, return the sum (reduction=sum)
    # or a list (reduction=None) of vector-matrix-vector product
    else:
        assert isinstance(x, Iterable)
        assert isinstance(y, Iterable)
        assert isinstance(m, Iterable)
        assert len(x) == len(y) == len(m)
        if reduce == 'sum':
            return sum(tf.reduce_sum(tf.einsum('bi,bij,bj->b', xx, mm, yy)) for (xx, mm, yy) in zip(x, m, y))
        elif reduce is None:
            return [tf.einsum('bi,bij,bj->b', xx, mm, yy) for (xx, mm, yy) in zip(x, m, y)]


def bmm(x, y):
    """
    batch matrix multiplication
    """
    return tf.matmul(x, y)


def pdist2(x, y, epsilon=1e-4):
    """
    pairwise distance between x and y in batch
    """
    _, n, d = x.shape
    m = y.shape[1]
    dtype = x.dtype
    epsilon = tf.convert_to_tensor(epsilon, dtype=dtype)
    return tf.sqrt(
            tf.einsum('bij, jk->bik', x ** 2, tf.ones((d, m), dtype=dtype)) +
            tf.einsum('ij, bkj->bik', tf.ones((n, d), dtype=dtype), y ** 2) -
            2 * tf.einsum('bij, bkj->bik', x, y) + epsilon) - tf.sqrt(epsilon)


def solve(A, b):
    """
    solution to linear equation Ax=b in batch
    """
    return tf.squeeze(tf.linalg.solve(A, tf.expand_dims(b, -1)), -1)


def gradient(gradient_tape, target, sources,
                output_gradients=None,
                unconnected_gradients=tf.UnconnectedGradients.ZERO):
    """
    compute the gradient with default value of unconnected_gradients=0
    """
    assert isinstance(gradient_tape, tf.GradientTape)
    return gradient_tape.gradient(
        target, sources,
        output_gradients=output_gradients,
        unconnected_gradients=unconnected_gradients
    )


def jacobian(gradient_tape, target, sources,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
                parallel_iterations=None,
                experimental_use_pfor=True):
    """
    compute the jacobian with default value of unconnected_gradients=0
    """
    assert isinstance(gradient_tape, tf.GradientTape)
    return gradient_tape.jacobian(
        target, sources,
        unconnected_gradients=unconnected_gradients,
        parallel_iterations=parallel_iterations,
        experimental_use_pfor=experimental_use_pfor
    )


def batch_jacobian(gradient_tape, target, source,
                      unconnected_gradients=tf.UnconnectedGradients.ZERO,
                      parallel_iterations=None,
                      experimental_use_pfor=True):
    """
    compute the batch_gradient with default value of unconnected_gradients=0
    """
    assert isinstance(gradient_tape, tf.GradientTape)
    return gradient_tape.batch_jacobian(target, source,
                                        unconnected_gradients=unconnected_gradients,
                                        parallel_iterations=parallel_iterations,
                                        experimental_use_pfor=experimental_use_pfor)


class MLP(tf.Module):
    def __init__(self, x_shape, y_shape, hidden_units=(), activation='tanh',
        hidden_layer_init_scale=1.0, output_layer_init_scale=1.0, init_distribution='uniform',
        bias_initializer='zeros', dtype=tf.float32, name='mlp'):

        super(MLP, self).__init__(name=name)

        self.layers = []
        in_dim = x_shape
        with self.name_scope:
            for h in hidden_units:
                kernel_initializer = tf.keras.initializers.VarianceScaling(
                    mode='fan_avg',
                    distribution=init_distribution,
                    scale=hidden_layer_init_scale**2
                    )
                layer = tf.keras.layers.Dense(
                            h, activation=activation,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            dtype=dtype)
                layer.build(in_dim)
                self.layers.append(layer)
                in_dim = h

            kernel_initializer = tf.keras.initializers.VarianceScaling(
                mode='fan_avg',
                distribution=init_distribution,
                scale=output_layer_init_scale**2
                )
            layer = tf.keras.layers.Dense(
                        y_shape,
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        dtype=dtype)
            layer.build(in_dim)
            self.layers.append(layer)

        self.units, self.activation = hidden_units, activation

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x