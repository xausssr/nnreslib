from __future__ import annotations

import collections.abc as ca
import functools
import operator
from enum import Enum, unique
from typing import Generator, Optional, Sequence, Tuple, Union, overload

from ..backend import graph as G
from ..backend.activation_functions import relu, sigmoid, softmax, tanh  # TODO: move to graph
from ..utils.serialized_types import SerializedShapeType


@unique
class ActivationFunctions(Enum):
    SIGMOID = functools.partial(sigmoid)
    RELU = functools.partial(relu)
    TANH = functools.partial(tanh)
    SOFT_MAX = functools.partial(softmax)

    @property
    def func(self) -> G.Tensor:
        return self.value.func  # pylint:disable=no-member


class Shape:
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, dim: Optional[Sequence[int]], *, is_null: bool = False) -> None:
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

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[int, Sequence[int]]:
        if isinstance(index, (int, slice)):
            return self.dimension[index]
        raise TypeError(f"index must be integer or slices, not {type(index).__name__}")

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

    def serialize(self) -> SerializedShapeType:
        if not self.is_null:
            return [*self.dimension]
        return dict(shape=[*self.dimension], is_null=self.is_null)
