from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Tuple, Type

import numpy as np

from .. import ForwardGraph
from ..graph import Graph
from ...architecture import Architecture
from ...backend import graph as G

_logger = logging.getLogger(__name__)


class FitGraph(Graph):
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
        train_dataset = self._prepare_batch(train_x_data, train_y_data, shuffle)
        valid_dataset = self._prepare_batch(valid_x_data, valid_y_data, shuffle)

        current_train_loss = 1e21
        epoch = 0

        while current_train_loss > min_error and epoch < max_epoch:
            epoch += 1
            current_train_loss = 0.0
            # TODO: move batch processing to function
            train_batches = self.session.run(self._get_batches(train_dataset))
            for batch in zip(train_batches[0], train_batches[1]):
                loss_on_batch, _, method_params = self._process_train_batch(batch, **kwargs)
                self._process_batch_result(method_params, kwargs)
                current_train_loss += loss_on_batch
                # TODO: calculate metrics for training data

            current_validation_loss = 0.0
            valid_batches = self.session.run(self._get_batches(valid_dataset))
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
