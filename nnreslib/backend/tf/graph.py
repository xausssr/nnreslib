from functools import partial

import tensorflow as tf

from .. import DTYPE

Dataset = tf.data.Dataset
GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
Operation = tf.Operation
PlaceholderType = tf.compat.v1.placeholder
Session = tf.compat.v1.Session
Tensor = tf.Tensor
TensorArray = partial(tf.TensorArray, dtype=DTYPE.value)
Variable = partial(tf.Variable, dtype=DTYPE.value)
VariableType = tf.Variable
assign = tf.compat.v1.assign
concat = tf.concat
conv2d = tf.nn.conv2d
eye = partial(tf.eye, dtype=DTYPE.value)
global_variables_initializer = tf.compat.v1.global_variables_initializer  # TODO: may be rename
gradients = tf.gradients
graph_function = tf.function
hessians = tf.hessians
inv = tf.linalg.inv
matmul = tf.matmul
max_pool = tf.nn.max_pool
multiply = tf.multiply
placeholder = partial(tf.compat.v1.placeholder, DTYPE.value)
reduce_mean = tf.reduce_mean
reshape = tf.reshape
split = tf.split
square = tf.square
squeeze = tf.squeeze
zeros = partial(tf.zeros, dtype=DTYPE.value)

__all__ = [
    "Dataset",
    "GradientDescentOptimizer",
    "Operation",
    "PlaceholderType",
    "Session",
    "Tensor",
    "TensorArray",
    "Variable",
    "VariableType",
    "assign",
    "concat",
    "conv2d",
    "eye",
    "global_variables_initializer",
    "gradients",
    "graph_function",
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
    "zeros",
]
