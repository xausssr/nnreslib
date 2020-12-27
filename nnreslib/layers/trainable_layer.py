from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from .base_layer import Layer
from ..utils.initialization import Initialization
from ..utils.merge import MergeInputs

if TYPE_CHECKING:
    import numpy as np

    from ..utils.types import ActivationFunction, Shape


class TrainableLayer(Layer):
    # pylint:disable=unused-argument
    def __init__(
        self,
        name: str,
        activation: ActivationFunction,
        merge: Optional[MergeInputs] = None,
        initializer: Initialization = Initialization(),
        is_out: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(name, merge=merge, is_out=is_out, **kwargs)
        self.initializer = initializer
        self.activation = activation
        self._weights: Optional[np.ndarray] = None
        self._biases: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def weights_shape(self) -> Shape:
        ...

    @property
    def weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError()
        return self._weights

    def set_weights(self, data_mean: float = 0.0, data_std: float = 0.0) -> None:
        self._weights = self.initializer.init_weights(self, data_mean, data_std)

    @property
    @abstractmethod
    def biases_shape(self) -> Shape:
        ...

    @property
    def biases(self) -> np.ndarray:
        if self._biases is None:
            raise ValueError()
        return self._biases

    def set_biases(self, data_mean: float = 0.0, data_std: float = 0.0) -> None:
        self._biases = self.initializer.init_biases(self, data_mean, data_std)
