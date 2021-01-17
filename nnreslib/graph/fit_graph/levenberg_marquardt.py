from typing import List

import numpy as np

from .fit_graph import FitGraph
from .. import ForwardGraph
from ..forward_graph import Parameters
from ...architecture import Architecture
from ...backend import graph as G


@FitGraph.register_fitter()
class LevenbergMarquardt(FitGraph):
    __slots__ = (
        "vector_error",
        "train_loss",
        "save_parameters",
        "restore_parameters",
        "save_hessian",
        "save_gradients",
        "regularization_factor",
        "parameters_update",
    )

    def __init__(self, batch_size: int, architecture: Architecture, forward_graph: ForwardGraph) -> None:
        super().__init__(batch_size, architecture, forward_graph)

        # FIXME: fix computation for multiple output layers
        self.vector_error = self.outputs - self.model_outputs  # use G.concat(G.flatten(x) for x in self.outputs)
        self.train_loss = G.reduce_mean(G.square(self.vector_error), name="Train_loss")

        # last_node = None  # FIXME: get correct last graph node
        # x = None # FIXME: input layers from forward_graph
        # XXX: self.grads_calculate = G.gradients(last_node, x)  # Need to care

        # Build computation graph for Levenberg-Marqvardt algorithm
        vectorized_parameters = LevenbergMarquardt._get_vectorized_parameters(forward_graph)

        parameters_store = G.Variable(G.zeros((architecture.neurons_count,)))
        self.save_parameters = G.assign(parameters_store, vectorized_parameters)
        self.restore_parameters = G.assign(vectorized_parameters, parameters_store)

        # TODO: Add Hessian approximation
        hessian_store = G.Variable(G.zeros((architecture.neurons_count, architecture.neurons_count)))
        hessian_approximation = G.hessians(self.train_loss, vectorized_parameters)[0]
        self.save_hessian = G.assign(hessian_store, hessian_approximation)

        gradients_store = G.Variable(G.zeros((architecture.neurons_count, 1)))
        gradients = self._get_gradients(vectorized_parameters, architecture.neurons_count)
        self.save_gradients = G.assign(gradients_store, gradients)

        self.regularization_factor = G.placeholder(shape=[1], name="regularization_factor")  # mu

        parameters_derivation = LevenbergMarquardt._get_parameters_derivation(
            architecture.neurons_count,
            hessian_store,
            self.regularization_factor,
            gradients_store,
            forward_graph.parameters,
        )

        self.parameters_update = LevenbergMarquardt._apply_parameters_derivation(
            forward_graph.parameters, parameters_derivation
        )

    @staticmethod
    def _get_vectorized_parameters(forward_graph: ForwardGraph) -> G.Tensor:
        """
        Convert list of weights and biases into one flat layer:

        [(w1, b1), ... (wN, bN)] -> flat(w1) + flat(b1) + ... + flat(wN) + flat(bN)

        Where plus means concatenation
        """

        return G.concat([G.reshape(y, [-1]) for x in forward_graph.parameters for y in x], 0)

    def _get_gradients(self, vectorized_parameters: G.Tensor, neurons_count: int) -> G.Tensor:
        return G.reshape(
            -G.gradients(self.train_loss, vectorized_parameters)[0],
            shape=(neurons_count, 1),
        )

    # pylint:disable=protected-access
    @staticmethod
    def _get_parameters_derivation(
        neurons_count: int,
        hessian_store: G.VariableType,
        regularization_factor: G.PlaceholderType,
        gradients_store: G.VariableType,
        parameters: List[Parameters],
    ) -> List[G.Tensor]:
        "Calculate dx and split it for individual parameters: dx -> [w0, b0, ..., wN, bN]"

        regularization_matrix = G.eye(neurons_count)
        return G.split(  # type: ignore
            G.squeeze(
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
            ),
            [np.prod(parameter.shape) for weights_biases in parameters for parameter in weights_biases],
            0,
        )

    # pylint:enable=protected-access

    @staticmethod
    def _apply_parameters_derivation(
        parameters: List[Parameters], parameters_derivation: List[G.Tensor]
    ) -> G.Operation:
        opt = G.GradientDescentOptimizer(learning_rate=1)
        parameters_update = []
        layer: Parameters
        for layer_num, layer in enumerate(parameters):
            parameters_update.append(
                (-G.reshape(parameters_derivation[layer_num * 2], layer.weights.shape), layer.weights)
            )
            parameters_update.append(
                (-G.reshape(parameters_derivation[layer_num * 2 + 1], layer.biases.shape), layer.biases)
            )
        return opt.apply_gradients(parameters_update)

    # TODO: describe method of LM
    def fit(self) -> None:
        ...
