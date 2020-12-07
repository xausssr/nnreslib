import math
from abc import ABC

import attr

from .types import ActivationFunction, Shape


@attr.s(auto_attribs=True)
class Layer(ABC):
    name: str

    def make_out_shape(self, inputs: Shape, pad: Shape) -> Shape:  # pylint:disable=unused-argument,no-self-use
        return Shape()

    def check_output_shape(self) -> bool:  # pylint:disable=no-self-use
        return True


@attr.s(auto_attribs=True)
class BaseConvolutionMaxPoolLayer(Layer):
    kernel: Shape
    stride: Shape

    def make_out_shape(self, inputs: Shape, pad: Shape) -> Shape:
        return Shape(
            *(
                math.ceil((pre_layer_dim - kernel + 2 * pad) / stride + 1)
                for pre_layer_dim, kernel, pad, stride in zip(inputs, self.kernel, pad, self.stride)
            )
        )


@attr.s(auto_attribs=True)
class ConvolutionLayer(BaseConvolutionMaxPoolLayer):
    filters: int
    pad: Shape
    activation: ActivationFunction
    out_shape: Shape = attr.ib(default=Shape())

    def check_output_shape(self) -> bool:
        return all(self.out_shape)


@attr.s(auto_attribs=True)
class MaxPoolLayer(BaseConvolutionMaxPoolLayer):
    out_shape: Shape = attr.ib(default=Shape())

    def check_output_shape(self) -> bool:  # TODO: implement
        raise NotImplementedError()


@attr.s(auto_attribs=True)
class FlattenLayer(Layer):
    out_shape: Shape = attr.ib(default=Shape())

    def check_output_shape(self) -> bool:  # TODO: implement
        raise NotImplementedError()


@attr.s(auto_attribs=True)
class FullyConnectedLayer(Layer):
    neurons: int
    activation: ActivationFunction
