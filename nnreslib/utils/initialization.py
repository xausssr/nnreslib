from __future__ import annotations

import collections.abc as ca
import functools
import math
from enum import Enum
from typing import TYPE_CHECKING, Callable, Union

import numpy as np

from .types import Shape
from ..utils.serialized_types import SerializedInitializationType, SerializedInitializeFunctionType

if TYPE_CHECKING:
    from ..layers import TrainableLayer


# pylint:disable=unused-argument
def _zeros(
    input_shape: Shape, output_shape: Shape, parameter_shape: Shape, data_mean: float = 0.0, data_std: float = 0.0
) -> np.ndarray:
    return np.zeros(parameter_shape.dimension)


def _he_normal(
    input_shape: Shape, output_shape: Shape, parameter_shape: Shape, data_mean: float = 0.0, data_std: float = 0.0
) -> np.ndarray:
    border = math.sqrt(6) / (math.sqrt(input_shape.prod + output_shape.prod))
    return np.random.uniform(-1 * border, border, size=parameter_shape.dimension)


def _haykin(
    input_shape: Shape, output_shape: Shape, parameter_shape: Shape, data_mean: float, data_std: float
) -> np.ndarray:
    """
    This activation use statistic from training data: mean, std and length of dataset
    """

    def rsqrt(value: float) -> float:
        return 1 / math.sqrt(value)

    def sqr(value: float) -> float:
        return math.pow(value, 2)

    border = 4 * math.sqrt(3) * rsqrt(input_shape.prod) * rsqrt(sqr(data_mean) + sqr(data_std))

    return np.random.uniform(-1 * border, border, size=parameter_shape.dimension)


class StandartInitializer(Enum):
    ZEROS = functools.partial(_zeros)
    HE_NORMAL = functools.partial(_he_normal)
    HAYKIN = functools.partial(_haykin)

    @property
    def func(self) -> InitializerType:
        return self.value.func  # pylint:disable=no-member


InitializerType = Callable[[Shape, Shape, Shape, float, float], np.ndarray]


class Initialization:
    # FIXME: write docstring
    def __init__(
        self,
        weights_initializer: Union[StandartInitializer, InitializerType] = StandartInitializer.HE_NORMAL,
        biases_initializer: Union[StandartInitializer, InitializerType] = StandartInitializer.ZEROS,
    ) -> None:
        def check_initializer(initializer: Union[StandartInitializer, InitializerType]) -> InitializerType:
            if isinstance(initializer, StandartInitializer):
                return initializer.func
            if isinstance(initializer, ca.Callable):  # type: ignore
                return initializer
            raise ValueError("Pass callable objects to __init__")

        self.weights = check_initializer(weights_initializer)
        self.biases = check_initializer(biases_initializer)

    def init_weights(self, layer: TrainableLayer, data_mean: float = 0.0, data_std: float = 0.0) -> np.ndarray:
        return self.weights(layer.input_shape, layer.output_shape, layer.weights_shape, data_mean, data_std)

    def init_biases(self, layer: TrainableLayer, data_mean: float = 0.0, data_std: float = 0.0) -> np.ndarray:
        return self.biases(layer.input_shape, layer.output_shape, layer.biases_shape, data_mean, data_std)

    def serialize(self) -> SerializedInitializationType:
        def serialize_initializer(initializer: InitializerType) -> SerializedInitializeFunctionType:
            for std_initializer in StandartInitializer:
                if std_initializer.func == initializer:
                    return std_initializer.name
            return dict(function="custom")  # XXX: support serialize CustomInitializer

        return dict(
            weights_initializer=serialize_initializer(self.weights),
            biases_initializer=serialize_initializer(self.biases),
        )
