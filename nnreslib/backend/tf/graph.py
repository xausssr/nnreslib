from functools import partial

import tensorflow as tf

from .. import DTYPE

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
Session = tf.compat.v1.Session
Tensor = tf.Tensor
PlaceholderType = tf.compat.v1.placeholder
VariableType = tf.Variable
assign = tf.compat.v1.assign
conv2d = tf.nn.conv2d
eye = tf.eye
global_variables_initializer = tf.compat.v1.global_variables_initializer  # TODO: may be rename
gradients = tf.compat.v1.gradients
gradients = tf.gradients
hessians = tf.hessians
inv = tf.linalg.inv
matmul = tf.matmul
max_pool = tf.nn.max_pool
multiply = tf.multiply
placeholder = partial(tf.compat.v1.placeholder, DTYPE.value)
reduce_mean = tf.reduce_mean
reshape = tf.reshape
split = tf.split  # TODO: check is need
square = tf.square
squeeze = tf.squeeze
variable = partial(tf.Variable, dtype=DTYPE.value)
zeros = partial(tf.zeros, dtype=DTYPE.value)

__all__ = [
    "GradientDescentOptimizer",
    "Session",
    "Tensor",
    "PlaceholderType",
    "VariableType",
    "assign",
    "conv2d",
    "eye",
    "global_variables_initializer",
    "gradients",
    "hessians",
    "inv",
    "matmul",
    "max_pool",
    "multiply",
    "placeholder",
    "reduce_mean",
    "reshape",
    "split",
    "square",
    "squeeze",
    "variable",
    "zeros",
]
