from __future__ import annotations

from typing import Optional

from .base_conv_layer import BaseConvLayer
from .base_layer import Layer
from .trainable_layer import TrainableLayer
from ..utils.initialization import Initialization
from ..utils.merge import MergeInputs
from ..utils.types import ActivationFunctions, Shape


class ConvolutionLayer(BaseConvLayer, TrainableLayer):
    __slots__ = ("filters", "pad")

    def __init__(
        self,
        name: str,
        kernel: Shape,
        stride: Shape,
        filters: int,
        pad: Optional[Shape] = None,
        merge: Optional[MergeInputs] = None,
        activation: ActivationFunctions = ActivationFunctions.RELU,
        initializer: Initialization = Initialization(),
        is_out: bool = False,
    ) -> None:
        """
        #TODO: describe default values for layer
        @xausssr
        """
        super().__init__(
            name=name,
            merge=merge,
            is_out=is_out,
            activation=activation,
            initializer=initializer,
            kernel=kernel,
            stride=stride,
        )
        if filters < 1:
            raise ValueError("'filters' must be greater than 0")
        self.filters = filters
        if pad is None:
            pad = Shape(*((0,) * len(self.kernel)), is_null=True)
        self.pad = pad

    def _set_output_shape(self) -> None:
        self._output_shape = self._calc_output_shape(2 * self.pad, self.filters)

    @property
    def neurons_count(self) -> int:
        return self.kernel.prod * self.input_shape[-1] * self.filters + self.filters

    @property
    def weights_shape(self) -> Shape:
        return Shape(*self.kernel, self.input_shape[-1], self.filters)

    @property
    def biases_shape(self) -> Shape:
        return Shape(
            *((1,) * len(self.kernel)), 1, self.filters
        )  # TODO: # Make third dim equal to self.input_shape[-1]


class MaxPoolLayer(BaseConvLayer):
    __slots__ = ()

    def _set_output_shape(self) -> None:
        self._output_shape = self._calc_output_shape()


class FlattenLayer(Layer):
    __slots__ = ()

    def _set_output_shape(self) -> None:
        self._output_shape = Shape(self.input_shape.prod)


class FullyConnectedLayer(TrainableLayer):
    __slots__ = ("neurons",)

    def __init__(
        self,
        name: str,
        neurons: int,
        merge: Optional[MergeInputs] = None,
        activation: ActivationFunctions = ActivationFunctions.SIGMOID,
        initializer: Initialization = Initialization(),
        is_out: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            merge=merge,
            is_out=is_out,
            activation=activation,
            initializer=initializer,
        )
        if neurons < 1:
            raise ValueError("'neurons' must be greater than 0")
        self.neurons = neurons

    @staticmethod
    def _check_input_shape(value: Shape) -> bool:
        return len(value) == 1

    def _set_output_shape(self) -> None:
        self._output_shape = Shape(self.neurons)

    @property
    def neurons_count(self) -> int:
        return self.input_shape[0] * self.neurons + self.neurons

    @property
    def weights_shape(self) -> Shape:
        return Shape(self.input_shape[0], self.neurons)

    @property
    def biases_shape(self) -> Shape:
        return Shape(1, self.neurons)
