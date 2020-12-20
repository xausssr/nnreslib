# TODO: may be need split this layers by types
from __future__ import annotations

from typing import Any

from .base2d_layer import Base2DLayer
from .base_layer import Layer
from .trainable_layer import TrainableLayer
from ..utils.initialization import Initialization
from ..utils.types import ActivationFunction, MergeFunction, Shape


class ConvolutionLayer(Base2DLayer, TrainableLayer):
    __slots__ = ("filters", "pad")

    def __init__(
        self,
        name: str,
        kernel: Shape,
        stride: Shape,
        filters: int,
        pad: Shape = Shape(0, 0, is_null=True),
        initializer: Initialization = Initialization(),
        activation: ActivationFunction = ActivationFunction.RELU,
        merge_func: MergeFunction = MergeFunction.PASSTHROUGH,
        is_out: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            merge_func=merge_func,
            is_out=is_out,
            initializer=initializer,
            activation=activation,
            kernel=kernel,
            stride=stride,
        )
        if filters < 1:
            raise ValueError("'filter' must be greater than 0")
        self.filters = filters
        self.pad = pad

    def set_output_shape(self, **kwargs: Any) -> None:
        self._output_shape = self._set_output_shape(2 * self.pad)

    def get_last_filters_count(self, last_filters_count: int = 0) -> int:
        return self.filters

    @property
    def neurons_count(self) -> int:
        return self.kernel.prod * self.input_shape[-1] * self.filters + self.filters

    @property
    def weights_shape(self) -> Shape:
        return Shape(*self.kernel, self.input_shape[-1], self.filters)

    @property
    def biases_shape(self) -> Shape:
        return Shape(1, 1, 1, self.filters)


class MaxPoolLayer(Base2DLayer):
    __slots__ = ()

    def set_output_shape(self, **kwargs: Shape) -> None:
        self._output_shape = self._set_output_shape(Shape(0, 0, is_null=True))


class FlattenLayer(Layer):
    __slots__ = ()
    OUTPUT_SHAPE_PARAM = "last_filters_count"

    def set_output_shape(self, **kwargs: Any) -> None:
        self._output_shape = Shape(self.input_shape.prod * kwargs[FlattenLayer.OUTPUT_SHAPE_PARAM])


class FullyConnectedLayer(TrainableLayer):
    __slots__ = ("neurons",)

    def __init__(
        self,
        name: str,
        neurons: int,
        initializer: Initialization = Initialization(),
        activation: ActivationFunction = ActivationFunction.SIGMOID,
        merge_func: MergeFunction = MergeFunction.PASSTHROUGH,
        is_out: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            merge_func=merge_func,
            is_out=is_out,
            initializer=initializer,
            activation=activation,
        )
        if neurons < 1:
            raise ValueError("'neurons' must be greater than 0")
        self.neurons = neurons

    @staticmethod
    def _check_input_shape(value: Shape) -> bool:
        return len(value) == 1

    @property
    def neurons_count(self) -> int:
        return self.input_shape[0] * self.neurons + self.neurons

    @property
    def weights_shape(self) -> Shape:
        return Shape(self.input_shape[0], self.neurons)

    @property
    def biases_shape(self) -> Shape:
        return Shape(1, self.neurons)
