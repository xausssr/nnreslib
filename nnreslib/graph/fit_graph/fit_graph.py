from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, Tuple, Type, Union

import numpy as np

from .. import ForwardGraph
from ..graph import Graph
from ...architecture import Architecture
from ...backend import graph as G
from ...utils.metrics import Metrics, OpMode

_logger = logging.getLogger(__name__)


class FitGraph(Graph):
    _fitters: Dict[str, Type[FitGraph]] = {}

    __slots__ = ("architecture", "forward_graph", "outputs", "model_outputs")

    def __init__(
        self,
        batch_size: int,
        architecture: Architecture,
        forward_graph: ForwardGraph,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(batch_size)
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
        self, batch_x: np.ndarray, batch_y: Iterable[np.ndarray], **kwargs: Any
    ) -> Tuple[float, Tuple[np.ndarray, ...], Tuple[Any, ...]]:
        ...

    @abstractmethod
    def _process_valid_batch(
        self, batch_x: np.ndarray, batch_y: Iterable[np.ndarray]
    ) -> Tuple[float, Tuple[np.ndarray, ...]]:
        ...

    @abstractmethod
    def _process_batch_result(self, params: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        ...

    def fit(
        self,
        train_x_data: np.ndarray,  # TODO: x may be list...
        train_y_data: Union[np.ndarray, Iterable[np.ndarray]],
        valid_x_data: np.ndarray,  # TODO: valid data is optional
        valid_y_data: Union[np.ndarray, Iterable[np.ndarray]],
        metrics: Metrics,
        max_epoch: int = 100,
        min_error: float = 1e-10,
        shuffle: bool = True,
        logging_step: int = 10,
        **kwargs: Any,
    ) -> Tuple[int, float]:
        _logger.info("Start training")
        train_dataset = self._prepare_batch(train_x_data, train_y_data, shuffle)
        valid_dataset = self._prepare_batch(valid_x_data, valid_y_data, shuffle)

        current_train_loss = 1e21
        epoch = 0

        while current_train_loss > min_error and epoch < max_epoch:
            epoch += 1
            # TODO: move batch processing to function
            with metrics.batch_metrics(OpMode.TRAIN, epoch) as batch_metrics:
                current_train_loss = 0.0
                for batch_x, batch_y in self._get_batches(train_dataset, True):
                    loss_on_batch, prediction, method_params = self._process_train_batch(batch_x, batch_y, **kwargs)
                    self._process_batch_result(method_params, kwargs)
                    current_train_loss += loss_on_batch
                    batch_metrics.calc_batch(batch_y, prediction)

            # TODO: make validation on validate_step
            with metrics.batch_metrics(OpMode.VALID, epoch) as batch_metrics:
                current_validation_loss = 0.0
                for batch_x, batch_y in self._get_batches(valid_dataset, True):
                    loss_on_batch, prediction = self._process_valid_batch(batch_x, batch_y)
                    current_validation_loss += loss_on_batch
                    batch_metrics.calc_batch(batch_y, prediction)

            # TODO: plot it
            current_train_loss = current_train_loss / self.session.run(train_dataset.cardinality())
            current_validation_loss = current_validation_loss / self.session.run(valid_dataset.cardinality())

            if epoch % logging_step == 0:
                _logger.info(
                    "epoch %7d: train error: %.12f; validation error: %.12f",
                    epoch,
                    current_train_loss,
                    current_validation_loss,
                )
        _logger.warning("Train ended on epoch: %s  with loss: %.12f", epoch, current_train_loss)
        return epoch, current_train_loss
