from functools import partial

import tensorflow as tf

from .. import DTYPE

Adadelta = tf.compat.v1.train.AdadeltaOptimizer
Adagrad = tf.compat.v1.train.AdagradOptimizer
Adam = tf.compat.v1.train.AdamOptimizer
DType = tf.as_dtype(DTYPE.value)
Dataset = tf.compat.v1.data.Dataset
GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
Momentum = tf.compat.v1.train.MomentumOptimizer
Operation = tf.Operation
OutOfRangeError = tf.errors.OutOfRangeError
PlaceholderType = tf.compat.v1.placeholder
RMSProp = tf.compat.v1.train.RMSPropOptimizer
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
losses_mse = tf.compat.v1.losses.mean_squared_error
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
    "Adadelta",
    "Adagrad",
    "Adam",
    "DType",
    "Dataset",
    "GradientDescentOptimizer",
    "Momentum",
    "Operation",
    "OutOfRangeError",
    "PlaceholderType",
    "RMSProp",
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
    "losses_mse",
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
