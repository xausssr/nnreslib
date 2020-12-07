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
            if dim < 1:
                raise ValueError("Empty or negative dimension")

    def __mul__(self, mul: int) -> "Shape":
        return type(self)(*(dim * mul for dim in self.dimension))

    __rmul__ = __mul__

    def __iter__(self) -> Generator[int, None, None]:
        for dim in self.dimension:
            yield dim

    def __getitem__(self, index: int) -> int:
        return self.dimension[index]

    @property
    def prod(self) -> int:
        if not self.dimension:
            return 0
        return functools.reduce(operator.mul, self.dimension, 1)

    def __str__(self) -> str:
        dim_print = "x".join(str(dim) for dim in self.dimension)
        return f"Shape: {dim_print}"

    __repr__ = __str__

    def __eq__(self, shape: object) -> bool:
        if isinstance(shape, Shape):
            return self.dimension == shape.dimension
        return False
