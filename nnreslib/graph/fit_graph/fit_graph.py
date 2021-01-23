from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple, Type

from .. import ForwardGraph
from ...architecture import Architecture
from ...backend import graph as G

if TYPE_CHECKING:
    import numpy as np

_logger = logging.getLogger(__name__)


class FitGraph(ABC):
    _fitters: Dict[str, Type[FitGraph]] = {}

    __slots__ = ("batch_size", "architecture", "forward_graph", "outputs", "model_outputs", "session")

    def __init__(
        self,
        batch_size: int,
        architecture: Architecture,
        forward_graph: ForwardGraph,  # pylint: disable=unused-argument
    ) -> None:
        self.batch_size = batch_size
        self.architecture = architecture
        self.forward_graph = forward_graph
        self.outputs = G.squeeze(
            G.concat(
                [
                    G.reshape(G.placeholder(name=x.name, shape=(batch_size, *x.output_shape)), [-1])
                    for x in architecture.output_layers
                ],
                0,
            )
        )
        self.model_outputs = G.squeeze(G.concat([G.reshape(x, [-1]) for x in forward_graph.outputs], 0))
        self.session = G.Session()

    @classmethod
    def register_fitter(cls) -> Callable[[Type[FitGraph]], Type[FitGraph]]:
        def decorator(fitter: Type[FitGraph]) -> Type[FitGraph]:
            cls._fitters[fitter.__name__] = fitter
            return fitter

        return decorator

    # XXX: implement call fit on fitter directly from this method (get_fitter -> fit)
    @classmethod
    def get_fitter(cls, fit_type: str) -> Type[FitGraph]:
        fitter = cls._fitters.get(fit_type)
        if fitter:
            return fitter
        raise ValueError(f"Unsupported fit type: {fit_type}")

    @staticmethod
    def _prepare_batch(x_data: np.ndarray, y_data: np.ndarray, batch_size: int, shuffle: bool = True) -> G.Dataset:
        x_dataset = G.Dataset.from_tensor_slices(x_data)
        y_dataset = G.Dataset.from_tensor_slices(y_data)
        dataset = G.Dataset.zip((x_dataset, y_dataset))
        if shuffle:
            dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)
        return dataset.batch(batch_size, drop_remainder=True)

    @staticmethod
    @G.graph_function  # type: ignore
    def _get_batches(data: G.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        tensor_array_x = G.TensorArray(size=0, dynamic_size=True)
        tensor_array_y = G.TensorArray(size=0, dynamic_size=True)
        i = 0
        for batch_x, batch_y in data:
            tensor_array_x = tensor_array_x.write(i, batch_x)
            tensor_array_y = tensor_array_y.write(i, batch_y)
            i += 1
        return tensor_array_x.stack(), tensor_array_y.stack()

    @abstractmethod
    def _process_train_batch(
        self, batch: Tuple[np.ndarray, np.ndarray], **kwargs: Any
    ) -> Tuple[float, np.ndarray, Tuple[Any, ...]]:
        ...

    @abstractmethod
    def _process_valid_batch(self, batch: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, np.ndarray]:
        ...

    @abstractmethod
    def _process_batch_result(self, params: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        ...

    def fit(
        self,
        train_x_data: np.ndarray,
        train_y_data: np.ndarray,
        valid_x_data: np.ndarray,  # TODO: valid data is optional
        valid_y_data: np.ndarray,
        max_epoch: int = 100,
        min_error: float = 1e-10,
        shuffle: bool = True,
        logging_step: int = 10,
        **kwargs: Any,
    ) -> Tuple[int, float]:
        train_dataset = FitGraph._prepare_batch(train_x_data, train_y_data, self.batch_size, shuffle)
        valid_dataset = FitGraph._prepare_batch(valid_x_data, valid_y_data, self.batch_size, shuffle)

        current_train_loss = 1e21
        epoch = 0

        while current_train_loss > min_error and epoch < max_epoch:
            epoch += 1
            current_train_loss = 0.0
            # TODO: move batch processing to function
            train_batches = self.session.run(FitGraph._get_batches(train_dataset))
            for batch in zip(train_batches[0], train_batches[1]):
                loss_on_batch, _, method_params = self._process_train_batch(batch, **kwargs)
                self._process_batch_result(method_params, kwargs)
                current_train_loss += loss_on_batch
                # TODO: calculate metrics for training data

            current_validation_loss = 0.0
            valid_batches = self.session.run(FitGraph._get_batches(valid_dataset))
            for batch in zip(valid_batches[0], valid_batches[1]):
                loss_on_batch, _ = self._process_valid_batch(batch)
                current_validation_loss += loss_on_batch
                # TODO: calculate metrics for validation data

            # TODO: plot it
            current_train_loss = current_train_loss / train_batches[0].shape[0]
            current_validation_loss = current_validation_loss / valid_batches[0].shape[0]

            if epoch % logging_step == 0:
                _logger.info(
                    "epoch %7d: train error: %.12f; validation error: %.12f",
                    epoch,
                    current_train_loss,
                    current_validation_loss,
                )

        _logger.warning("Train ended on epoch: %s  with loss: %.12f", epoch, current_train_loss)
        return epoch, current_train_loss
