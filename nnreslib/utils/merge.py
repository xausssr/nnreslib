from __future__ import annotations

import functools
from enum import Enum, unique
from typing import Callable, Optional, Sequence, Union

from .types import Shape
from ..backend import graph as G
from ..utils.serialized_types import SerializedMergeFunctionsType, SerializedMergeInputsType


def _data_merge_not_implemented(
    main_input: Union[Callable[..., G.Tensor]], *other_input: Union[Callable[..., G.Tensor]]
) -> Union[Callable[..., G.Tensor], G.Tensor]:
    raise NotImplementedError


@unique
class MergeFunctions(Enum):
    # TODO: fix annotation
    value: Union[Callable[..., G.Tensor], G.Tensor]
    RESHAPE_TO_MAIN = functools.partial(_data_merge_not_implemented)

    def serialize(self) -> SerializedMergeFunctionsType:
        return self.name

    @classmethod
    def deserialize(cls, data: SerializedMergeFunctionsType) -> MergeFunctions:
        return cls[data]


def _check_merged_tensors(main_input: Shape, *other_input: Shape, recurrent: Optional[Sequence[Shape]] = None) -> None:
    for shape in other_input:
        if len(shape) != len(main_input):
            # TODO: fix this by adding shadow layer
            raise ValueError("Input tensors have a different number of dimentions")
    if recurrent:
        _check_merged_tensors(main_input, *recurrent)


def _merge_flatten(main_input: Shape, *other_input: Shape, recurrent: Optional[Sequence[Shape]] = None) -> Shape:
    shape = main_input.prod + sum((x.prod for x in other_input))
    if recurrent:
        shape += sum((x.prod for x in recurrent))
    return Shape(shape)


def _merge_to_main(main_input: Shape, *other_input: Shape, recurrent: Optional[Sequence[Shape]] = None) -> Shape:
    if len(main_input) == 1:
        return _merge_flatten(main_input, *other_input, recurrent=recurrent)

    last_dim = sum((x[-1] for x in other_input)) + main_input[-1]
    if recurrent:
        last_dim += sum((x[-1] for x in recurrent))
    return Shape(*main_input[:-1], last_dim)


@unique
class MergeShapeFunction(Enum):
    RESHAPE_TO_MAIN = _merge_to_main


class MergeInputs:
    # FIXME: write docstring
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
        ...

    #     return self._merge_func.value(main_input, *other_input)        self,
    #     main_input: Union[Callable[..., G.Tensor], G.Tensor],
    #     *other_input: Union[Callable[..., G.Tensor], G.Tensor]
    # ) -> Union[Callable[..., G.Tensor], G.Tensor]:
    #     return self._merge_func.value(main_input, *other_input)

    def serialize(self) -> SerializedMergeInputsType:
        return dict(main_input=self.main_input, merge_func=self._merge_func.name)

    @classmethod
    def deserialize(cls, data: SerializedMergeInputsType) -> MergeInputs:
        return cls(**data)
