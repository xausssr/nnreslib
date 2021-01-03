from __future__ import annotations

import functools
from enum import Enum, unique
from typing import Callable, Optional, Union

from .types import Shape
from ..backend import graph as G


def _data_merge_not_implemented(
    main_input: Union[Callable[..., G.Tensor]], *other_input: Union[Callable[..., G.Tensor]]
) -> Union[Callable[..., G.Tensor], G.Tensor]:
    raise NotImplementedError


@unique
class MergeFunctions(Enum):
    # TODO: fix annotation
    value: Union[Callable[..., G.Tensor], G.Tensor]
    RESHAPE_TO_MAIN = functools.partial(_data_merge_not_implemented)


# pylint:disable=unused-argument
def _reshape_to_main(main_input: Shape, *other_input: Shape) -> Shape:
    return main_input


# pylint:enable=unused-argument


@unique
class MergeShapeFunction(Enum):
    RESHAPE_TO_MAIN = _reshape_to_main


class MergeInputs:
    __slots__ = ("main_input", "_merge_func", "result_shape")

    def __init__(self, main_input: str = "", merge_func: MergeFunctions = MergeFunctions.RESHAPE_TO_MAIN):
        self.main_input = main_input
        self._merge_func = merge_func
        self.result_shape: Optional[Shape] = None

    def calc_result_shape(self, main_input: Shape, *other_input: Shape) -> Shape:
        self.result_shape = getattr(MergeShapeFunction, self._merge_func.name)(main_input, *other_input)
        assert self.result_shape
        return self.result_shape

    # TODO: fix annotation
    def __call__(
        self,
        main_input: Union[Callable[..., G.Tensor], G.Tensor],
        *other_input: Union[Callable[..., G.Tensor], G.Tensor]
    ) -> Union[Callable[..., G.Tensor], G.Tensor]:
        return self._merge_func.value(main_input, *other_input)
