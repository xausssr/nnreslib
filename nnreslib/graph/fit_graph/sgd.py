from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .fit_graph import FitGraph, LossFunctionType
from .. import ForwardGraph
from ...architecture import Architecture
from ...backend import graph as G
from ...utils.types import LossFunctions


@FitGraph.register_fitter()
class SGD(FitGraph):
    __slots__ = (
        "optimizer",
        "loss",
        "train_loss",
        "learning_rate",
    )

    def __init__(
        self,
        batch_size: int,
        architecture: Architecture,
        forward_graph: ForwardGraph,
        loss: LossFunctionType = LossFunctions.MSE.value,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(batch_size, architecture, forward_graph, loss)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.session = forward_graph.session
        self.train_loss = loss(self.outputs, self.model_outputs)
        self.optimizer = G.GradientDescentOptimizer(self.learning_rate).minimize(self.train_loss)
        self.session.run(G.global_variables_initializer())

    def _process_train_batch(
        self, batch_x: np.ndarray, batch_y: Iterable[np.ndarray], **kwargs: Any
    ) -> Tuple[float, Tuple[np.ndarray, ...], Tuple[Any, ...]]:
        feed_dict = self._get_feed_dict(batch_x, batch_y)
        current_loss = self.session.run(self.train_loss, feed_dict)
        self.session.run(self.optimizer, feed_dict)
        return (
            current_loss,
            self.session.run(self.forward_graph.outputs, feed_dict),
            (),
        )

    def _process_valid_batch(
        self, batch_x: np.ndarray, batch_y: Iterable[np.ndarray]
    ) -> Tuple[float, Tuple[np.ndarray, ...]]:
        feed_dict = self._get_feed_dict(batch_x, batch_y)
        current_loss = self.session.run(self.train_loss, feed_dict)
        output = self.session.run(self.forward_graph.outputs, feed_dict)
        return current_loss, output
