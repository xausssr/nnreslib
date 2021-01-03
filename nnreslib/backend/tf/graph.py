from functools import partial

import tensorflow as tf

from .. import DTYPE

placeholder = partial(tf.compat.v1.placeholder, DTYPE.value)
_placeholder = tf.compat.v1.placeholder
variable = partial(tf.Variable, dtype=DTYPE.value)
_variable = tf.Variable
split = tf.split  # TODO: check is need
conv2d = tf.nn.conv2d
max_pool = tf.nn.max_pool
reshape = tf.reshape
matmul = tf.matmul
multiply = tf.multiply
squeeze = tf.squeeze
reduce_mean = tf.reduce_mean
square = tf.square
gradients = tf.compat.v1.gradients
gradients = tf.gradients
GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
eye = tf.eye
hessians = tf.hessians
zeros = partial(tf.zeros, dtype=DTYPE.value)
assign = tf.compat.v1.assign
inv = tf.linalg.inv
Session = tf.compat.v1.Session
global_variables_initializer = tf.compat.v1.global_variables_initializer  # TODO: may be rename
Tensor = tf.Tensor

__all__ = [
    "placeholder",
    "_placeholder",
    "variable",
    "_variable",
    "split",
    "conv2d",
    "max_pool",
    "reshape",
    "matmul",
    "multiply",
    "squeeze",
    "reduce_mean",
    "square",
    "gradients",
    "gradients",
    "GradientDescentOptimizer",
    "eye",
    "hessians",
    "zeros",
    "assign",
    "inv",
    "Session",
    "global_variables_initializer",
    "Tensor",
]
