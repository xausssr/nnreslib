import math
from abc import ABC

import attr

from .types import ActivationFunction, Shape


@attr.s(auto_attribs=True)
class Layer(ABC):
    name: str

    def make_out_shape(self, inputs: Shape, pad: Shape) -> Shape:  # pylint:disable=unused-argument,no-self-use
        return Shape()


@attr.s(auto_attribs=True)
class Base2DLayer(Layer):
    """
    @xausssr please retype doc-string
    """

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
class ConvolutionLayer(Base2DLayer):
    filters: int
    pad: Shape
    activation: ActivationFunction
    out_shape: Shape = attr.ib(default=Shape())


@attr.s(auto_attribs=True)
class MaxPoolLayer(Base2DLayer):
    out_shape: Shape = attr.ib(default=Shape())


@attr.s(auto_attribs=True)
class FlattenLayer(Layer):
    out_shape: Shape = attr.ib(default=Shape())

    def make_out_shape(self, inputs: Shape, pad: Shape) -> Shape:
        return Shape(inputs.prod)


@attr.s(auto_attribs=True)
class FullyConnectedLayer(Layer):
    neurons: int
    activation: ActivationFunction
