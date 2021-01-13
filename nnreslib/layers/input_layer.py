from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from .base_layer import Layer

if TYPE_CHECKING:
    from ..utils.types import Shape


class InputLayer(Layer):
    def __init__(self, name: str, input_shape: Shape):
        super().__init__(name)
        self._input_shape = input_shape
        self._output_shape = input_shape

    def _set_output_shape(self) -> None:
        self._output_shape = self._input_shape

    def set_shapes(self, main_input: Shape, *other_input: Shape) -> None:
        return

    # TODO: fix return type annotation
    def serialize(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            type=self.__class__.__name__,
            input_shape=self.input_shape.serialize(),
            output_shape=self.output_shape.serialize(),
        )
