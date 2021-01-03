from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type, Union

from ..backend import graph as G
from ..layers import ConvolutionLayer, FlattenLayer, FullyConnectedLayer, InputLayer, Layer, MaxPoolLayer

ArgsType = Union[int, G.Tensor]


# pylint:disable=unused-argument
class LayerFunc(ABC):
    _subclasses: Dict[Type[Layer], Type[LayerFunc]] = {}

    def __init__(self, layer: Layer, *args: ArgsType, **kwargs: ArgsType) -> None:
        ...

    @classmethod
    def register_subclass(cls) -> Callable[[Type[LayerFunc]], Type[LayerFunc]]:
        def decorator(subclass: Type[LayerFunc]) -> Type[LayerFunc]:
            try:
                subclass_type = subclass.__init__.__annotations__["layer"]
            except (KeyError, AttributeError) as exp:
                raise TypeError("Incorrect class for decorating") from exp
            cls._subclasses[subclass_type] = subclass
            return subclass

        return decorator

    # pylint:disable=protected-access
    @classmethod
    def get_layer_func(cls, layer: Layer, *args: ArgsType, **kwargs: ArgsType) -> Union[LayerFunc, G._placeholder]:
        if isinstance(layer, InputLayer):
            return G.placeholder(name=layer.name, shape=(kwargs["batch_size"], *layer.output_shape))
        layer_func = cls._subclasses.get(type(layer), None)
        if layer_func:
            return layer_func(layer, *args, **kwargs)
        raise ValueError(f"Unsupported type: {type(layer)}")

    # pylint:enable=protected-access

    @abstractmethod
    def __call__(self, layer_input: G.Tensor) -> G.Tensor:
        ...


# pylint:disable=super-init-not-called
@LayerFunc.register_subclass()
class ConvolutionFunc(LayerFunc):
    def __init__(
        self, layer: ConvolutionLayer, weights: G.Tensor, biases: G.Tensor, *args: ArgsType, **kwargs: ArgsType
    ) -> None:
        self.layer = layer
        self.weights = weights
        self.biases = biases

    def __call__(self, layer_input: G.Tensor) -> G.Tensor:
        pad = self.layer.pad / 2
        return self.layer.activation.value(
            G.conv2d(  # TODO: check layer dimension, and choice the right convolution
                layer_input,
                self.weights,
                (1, *self.layer.stride, 1),  # TODO: feature request
                padding=(0, 0, *pad, 0, 0),  # TODO: feature request
            )
            + self.biases
        )


@LayerFunc.register_subclass()
class MaxPoolFunc(LayerFunc):
    def __init__(self, layer: MaxPoolLayer, *args: ArgsType, **kwargs: ArgsType) -> None:
        self.layer = layer

    def __call__(self, layer_input: G.Tensor) -> G.Tensor:
        return G.max_pool(
            layer_input,
            (1, *self.layer.kernel, 1),
            (1, *self.layer.stride, 1),
            padding="SAME",  # TODO: need fix??? What about backend???
        )


@LayerFunc.register_subclass()
class FlattenFunc(LayerFunc):
    def __init__(self, layer: FlattenLayer, batch_size: int, *args: ArgsType, **kwargs: ArgsType) -> None:
        self.layer = layer
        self.batch_size = batch_size

    def __call__(self, layer_input: G.Tensor) -> G.Tensor:
        return G.reshape(
            layer_input,
            (self.batch_size, *self.layer.output_shape),
        )


@LayerFunc.register_subclass()
class FullyConnectedFunc(LayerFunc):
    def __init__(
        self, layer: FullyConnectedLayer, weights: G.Tensor, biases: G.Tensor, *args: ArgsType, **kwargs: ArgsType
    ) -> None:
        self.layer = layer
        self.weights = weights
        self.biases = biases

    def __call__(self, layer_input: G.Tensor) -> G.Tensor:
        return self.layer.activation.value(
            G.matmul(
                layer_input,
                self.weights,
            )
            + self.biases
        )


# pylint:enable=super-init-not-called
# pylint:enable=unused-argument
