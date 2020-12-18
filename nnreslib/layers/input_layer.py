from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .base_layer import Layer

if TYPE_CHECKING:
    import numpy as np

    from ..utils.types import Shape


class InputLayer(Layer):
    def __init__(self, name: str, input_shape: Shape, input_data: Optional[np.ndarray] = None):
        super().__init__(name)
        self._input_shape = input_shape
        self._output_shape = input_shape
        self.input_data = input_data
        self._input_mean: Optional[float] = None
        self._input_std: Optional[float] = None

    @property
    def input_mean(self) -> float:
        if self._input_mean is None:
            raise NotImplementedError
        return self._input_mean

    @property
    def input_std(self) -> float:
        if self._input_std is None:
            raise NotImplementedError
        return self._input_std

    def get_last_filters_count(self, last_filters_count: int = 0) -> int:
        return self.output_shape[-1]
