from abc import ABC, abstractmethod
from typing import Any, Optional

from ..utils.merge import MergeInputs
from ..utils.types import Shape


class Layer(ABC):
    # pylint:disable=unused-argument
    def __init__(
        self,
        name: str,
        merge: Optional[MergeInputs] = None,
        is_out: bool = False,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.merge = merge
        self.is_out = is_out
        self._input_shape: Optional[Shape] = None
        self._output_shape: Optional[Shape] = None

    @property
    def input_shape(self) -> Shape:
        if self._input_shape is None:
            raise ValueError("Call 'set_shapes' method at first")
        return self._input_shape

    @property
    def output_shape(self) -> Shape:
        if not self._output_shape:
            raise ValueError("Call 'set_shapes' method at first")
        return self._output_shape

    # pylint:disable=unused-argument
    @staticmethod
    def _check_input_shape(value: Shape) -> bool:
        return True

    @abstractmethod
    def _set_output_shape(self) -> None:
        self._output_shape = None

    def set_shapes(self, main_input: Shape, *other_input: Shape) -> None:
        if self.merge:
            input_shape = self.merge.calc_result_shape(main_input, *other_input)
        else:
            input_shape = main_input
        if not self._check_input_shape(input_shape):
            raise ValueError(f"Incorrect input shape for layer: {self.name}")
        self._input_shape = input_shape
        self._set_output_shape()

    @property
    def neurons_count(self) -> int:
        return 0
