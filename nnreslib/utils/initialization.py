from __future__ import annotations

import collections.abc as ca
import functools
import math
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np

from .types import Shape

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


InitializerType = Callable[[Shape, Shape, Shape, float, float], np.ndarray]


class Initialization:
    def __init__(
        self,
        weights_initializer: InitializerType = StandartInitializer.HE_NORMAL.value,
        biases_initializer: InitializerType = StandartInitializer.ZEROS.value,
    ) -> None:
        if not all(
            map(lambda x: isinstance(x, ca.Callable), (weights_initializer, biases_initializer))  # type: ignore
        ):
            raise ValueError("Pass callable objects to __init__")

        self.weights = weights_initializer
        self.biases = biases_initializer

    def init_weights(self, layer: TrainableLayer, data_mean: float = 0.0, data_std: float = 0.0) -> np.ndarray:
        return self.weights(layer.input_shape, layer.output_shape, layer.weights_shape, data_mean, data_std)

    def init_biases(self, layer: TrainableLayer, data_mean: float = 0.0, data_std: float = 0.0) -> np.ndarray:
        return self.biases(layer.input_shape, layer.output_shape, layer.biases_shape, data_mean, data_std)
