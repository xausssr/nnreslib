from abc import ABC
from typing import Any, Optional

from ..utils.types import Shape


class Layer(ABC):
    # pylint:disable=unused-argument
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self._input_shape: Optional[Shape] = None
        self._output_shape: Optional[Shape] = None

    @property
    def input_shape(self) -> Shape:
        if self._input_shape is None:
            raise ValueError("'input_shape' must be initialized")
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value: Shape) -> None:
        if self._check_input_shape(value):
            self._input_shape = value
        else:
            raise ValueError("Incorrect input shape")

    # pylint:disable=unused-argument
    @staticmethod
    def _check_input_shape(value: Shape) -> bool:
        return True

    @property
    def output_shape(self) -> Shape:
        if not self._output_shape:
            raise ValueError("Call 'fill_output_shape' method first")
        return self._output_shape

    # pylint:disable=unused-argument
    def set_output_shape(self, **kwargs: Any) -> None:
        self._output_shape = Shape()

    # pylint:disable=no-self-use
    def get_last_filters_count(self, last_filters_count: int = 0) -> int:
        return last_filters_count

    @property
    def neurons_count(self) -> int:
        return 0
