from __future__ import annotations

import collections.abc as ca
import functools
import operator
from enum import Enum
from typing import TYPE_CHECKING, overload

from .tf_helper import tf

if TYPE_CHECKING:
    from typing import Generator, Optional, Sequence, Tuple, Union


class ActivationFunction(Enum):
    SIGMOID = tf.nn.sigmoid
    RELU = tf.nn.relu
    TANH = tf.nn.tanh
    SOFT_MAX = tf.nn.softmax


class Shape:
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, dim: Optional[Sequence[int]]) -> None:
        ...

    @overload
    def __init__(self, dim: Optional[int], *dims: int, is_null: bool = False) -> None:
        ...

    # pylint:disable=keyword-arg-before-vararg
    def __init__(self, dim: Optional[Union[Sequence[int], int]] = None, *dims: int, is_null: bool = False) -> None:
        if dim is None:
            self.dimension: Tuple[int, ...] = ()
        elif isinstance(dim, ca.Sequence):
            self.dimension = (*dim,)
        else:
            self.dimension = (dim, *dims)
        self._prod = -1
        self.is_null = is_null
        for dimension in self.dimension:
            if not isinstance(dimension, int):
                raise ValueError("Shape arguments must be int")
            if not self.is_null and dimension < 1:
                raise ValueError("Empty or negative dimension")

    def __len__(self) -> int:
        return len(self.dimension)

    def __iter__(self) -> Generator[int, None, None]:
        for dim in self.dimension:
            yield dim

    def __getitem__(self, index: int) -> int:
        return self.dimension[index]

    def __mul__(self, value: object) -> Shape:
        if isinstance(value, int):
            return type(self)(*(dim * value for dim in self.dimension), is_null=self.is_null)
        raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(value).__name__}'")

    __rmul__ = __mul__

    def __truediv__(self, value: object) -> Shape:
        if isinstance(value, int):
            return type(self)(*(dim // value for dim in self.dimension), is_null=self.is_null)
        raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(value).__name__}'")

    def __str__(self) -> str:
        dim_print = "x".join(str(dim) for dim in self.dimension)
        return f"Shape: {dim_print}"

    def __repr__(self) -> str:
        dim_print = ", ".join(str(dim) for dim in self.dimension)
        if dim_print:
            dim_print += ", "
        dim_print += f"is_null={self.is_null}"
        return f"Shape({dim_print})"

    def __eq__(self, shape: object) -> bool:
        if isinstance(shape, Shape):
            return self.dimension == shape.dimension
        if isinstance(shape, (list, tuple)):
            return self.dimension == shape
        return False

    @property
    def prod(self) -> int:
        if not self.dimension:
            return 0
        if self._prod == -1:
            self._prod = functools.reduce(operator.mul, self.dimension, 1)
        return self._prod
