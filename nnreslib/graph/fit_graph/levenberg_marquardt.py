from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np

from .fit_graph import FitGraph, LossFunctionType
from .. import ForwardGraph
from ...architecture import Architecture
from ...backend import graph as G
from ...utils.types import LossFunctions


@FitGraph.register_fitter()
class LevenbergMarquardt(FitGraph):
    __slots__ = (
        "train_loss",
        "save_parameters",
        "restore_parameters",
        "save_hessian",
        "save_gradients",
        "regularization_factor",
        "parameters_update",
        "random_step",
        "parameters_random_update",
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
        # TODO fix issue: split graphs
        self.session = forward_graph.session
        # self.train_loss = G.losses_mse(self.outputs, self.model_outputs)

        self.train_loss = loss(self.outputs, self.model_outputs)

        # last_node = None  # FIXME: get correct last graph node
        # x = None # FIXME: input layers from forward_graph
        # XXX: self.grads_calculate = G.gradients(last_node, x)  # Need to care
        self.random_step = G.placeholder(shape=(architecture.neurons_count,), name="random_step")

        self.parameters_random_update = G.assign(
            self.forward_graph.parameters, self.forward_graph.parameters * self.random_step
        )
        # Build computation graph for Levenberg-Marqvardt algorithm
        parameters_store = G.Variable(G.zeros((architecture.neurons_count,)))
        self.save_parameters = G.assign(parameters_store, self.forward_graph.parameters)
        self.restore_parameters = G.assign(self.forward_graph.parameters, parameters_store)

        # TODO: Add Hessian approximation
        hessian_store = G.Variable(G.zeros((architecture.neurons_count, architecture.neurons_count)))

        hessian_approximation = G.hessians(self.train_loss, self.forward_graph.parameters)[0]
        self.save_hessian = G.assign(hessian_store, hessian_approximation)

        gradients_store = G.Variable(G.zeros((architecture.neurons_count, 1)))
        gradients = self._get_gradients(self.forward_graph.parameters, architecture.neurons_count)
        self.save_gradients = G.assign(gradients_store, gradients)

        self.regularization_factor = G.placeholder(shape=[], name="regularization_factor")  # mu

        parameters_derivation = LevenbergMarquardt._get_parameters_derivation(
            architecture.neurons_count,
            hessian_store,
            self.regularization_factor,
            gradients_store,
        )

        self.parameters_update = LevenbergMarquardt._apply_parameters_derivation(
            self.forward_graph.parameters, parameters_derivation
        )

        self.session.run(G.global_variables_initializer())

    def _get_gradients(self, vectorized_parameters: G.Tensor, neurons_count: int) -> G.Tensor:
        return G.reshape(
            -G.gradients(self.train_loss, vectorized_parameters)[0],
            shape=(neurons_count, 1),
        )

    @staticmethod
    def _get_parameters_derivation(
        neurons_count: int,
        hessian_store: G.VariableType,
        regularization_factor: G.PlaceholderType,
        gradients_store: G.VariableType,
    ) -> G.Tensor:
        regularization_matrix = G.eye(neurons_count)
        return G.squeeze(
            G.matmul(
                G.inv(
                    hessian_store
                    + G.multiply(
                        regularization_factor,
                        regularization_matrix,
                    ),
                ),
                gradients_store,
            ),
        )

    @staticmethod
    def _apply_parameters_derivation(parameters: G.Tensor, parameters_derivation: G.Tensor) -> G.Operation:
        opt = G.GradientDescentOptimizer(learning_rate=1)
        return opt.apply_gradients([(-parameters_derivation, parameters)])

    def _get_feed_dict(
        self, batch_x: np.ndarray, batch_y: Iterable[np.ndarray], **kwargs: Any
    ) -> Dict[Union[G.PlaceholderType, str], Union[float, np.ndarray]]:
        feed_dict = super()._get_feed_dict(batch_x, batch_y)

        regularisation_factor = kwargs.get("regularisation_factor")
        if regularisation_factor is not None:
            feed_dict[self.regularization_factor] = regularisation_factor

        return feed_dict

    # FIXME input data is List[np.ndarray] or np.ndarray
    def _process_train_batch(
        self, batch_x: np.ndarray, batch_y: Iterable[np.ndarray], **kwargs: Any
    ) -> Tuple[float, Tuple[np.ndarray, ...], Tuple[Any, ...]]:
        step_into_epoch: int = kwargs["step_into_epoch"]
        regularisation_factor_init: float = kwargs.get("regularisation_factor_init", 5.0)
        regularisation_factor: float = kwargs.get("regularisation_factor", regularisation_factor_init)
        regularisation_factor_decay: float = kwargs.get("regularisation_factor_decay", 10.0)
        regularisation_factor_increase: float = kwargs.get("regularisation_factor_increase", 10.0)
        percent_random: float = kwargs.get("percent_random", 0.2)

        feed_dict = self._get_feed_dict(batch_x, batch_y, regularisation_factor=regularisation_factor)

        make_random_step = False
        if feed_dict[self.regularization_factor] > G.DType.max / 10 ** (step_into_epoch + 1):
            feed_dict[self.random_step] = np.random.uniform(
                1.0 - percent_random, 1.0 + percent_random, size=(self.architecture.neurons_count,)
            )
            feed_dict[self.regularization_factor] = regularisation_factor_init
            make_random_step = True
            self.session.run(self.parameters_random_update, feed_dict)

        current_loss = self.session.run(self.train_loss, feed_dict)
        _, hess, _ = self.session.run([self.save_parameters, self.save_hessian, self.save_gradients], feed_dict)

        if make_random_step:
            feed_dict[self.random_step] = np.ones((self.architecture.neurons_count,))
            self.session.run(self.parameters_random_update, feed_dict)

        for step in range(step_into_epoch):
            self.session.run(self.parameters_update, feed_dict)
            new_loss = self.session.run(self.train_loss, feed_dict)
            if new_loss < current_loss:
                feed_dict[self.regularization_factor] /= regularisation_factor_decay
                break
            feed_dict[self.regularization_factor] *= regularisation_factor_increase
            if step != step_into_epoch - 1:
                self.session.run(self.restore_parameters)
        return (
            new_loss,
            self.session.run(self.forward_graph.outputs, feed_dict),
            (feed_dict[self.regularization_factor],),
        )

    def _process_batch_result(self, params: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        kwargs["regularisation_factor"] = params[0]

    def _process_valid_batch(
        self, batch_x: np.ndarray, batch_y: Iterable[np.ndarray]
    ) -> Tuple[float, Tuple[np.ndarray, ...]]:
        feed_dict = self._get_feed_dict(batch_x, batch_y)
        current_loss = self.session.run(self.train_loss, feed_dict)
        output = self.session.run(self.forward_graph.outputs, feed_dict)
        return current_loss, output
