import functools
import operator
from enum import Enum
from typing import Generator

from .tf_helper import tf


class ActivationFunction(Enum):
    SIGMOID = tf.nn.sigmoid
    RELU = tf.nn.relu
    TANH = tf.nn.tanh
    SOFT_MAX = tf.nn.softmax


class Shape:
    def __init__(self, *args: int):
        self.dimension = tuple([*args])
        for dim in self.dimension:
            if not isinstance(dim, int):
                raise ValueError("Shape arguments must be int")

    def __mul__(self, mul: int) -> "Shape":
        return type(self)(*(dim * mul for dim in self.dimension))

    __rmul__ = __mul__

    def __iter__(self) -> Generator[int, None, None]:
        for dim in self.dimension:
            yield dim

    def __getitem__(self, index: int) -> int:
        return self.dimension[index]

    def prod(self) -> int:
        return functools.reduce(operator.mul, self.dimension, 1)
