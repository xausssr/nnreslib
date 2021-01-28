from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .graph import Graph
from .layers import LayerFunc
from ..architecture import Architecture
from ..backend import graph as G
from ..layers import Layer, TrainableLayer

_logger = logging.getLogger(__name__)

# FIXME: Global fix G(TF) type annotations


class ForwardGraph(Graph):
    __slots__ = (
        "layers",
        "parameters",
        "inputs",
        "outputs",
        "_parameters_index"
        # "vector_error",
        # "train_loss",
    )

    def __init__(self, batch_size: int, architecture: Architecture) -> None:
        super().__init__(batch_size)
        self._parameters_index: Dict[str, int] = {}
        self.parameters = self._get_parameters_vector(architecture)

        _logger.info("Start building model graph")
        layers = self._create_layers(self.batch_size, architecture)

        # Move placeholders for input layers to graph layers dict
        self.layers: Dict[str, Union[Callable[..., G.Tensor], G.Tensor]] = {
            x: layers[x] for x in architecture._input_layers
        }  # FIXME: layer is Callable[..., G.Tensor]

        self.inputs = tuple([self.layers[x] for x in architecture._input_layers])

        # dict value is G.Variable
        recurrent_layers: Dict[str, G.Tensor] = {}

        # layers: Dict with partial defined layers: LayerFunc for layers, and placeholder for InputLayers.
        # Parameter for LayerFunc - input
        # self.layers: Dict with initialized layers: input is passed, really created graph's nodes
        # BUG: check architecture for InputLayer in middle of Model definition
        layer: Layer
        for layer, inputs in architecture:
            if layer.merge is None:
                self.layers[layer.name] = layers[layer.name](self.layers[inputs[0]])
            else:
                input_layers: List[G.Tensor] = []
                for input_layer in inputs:
                    if input_layer == layer.merge.main_input:
                        continue
                    # check if inputs has recurrent dependencies
                    # if so, create G.Variable(with layer shape and 0 init value) and mark recurrent layer
                    # Use this variable as argument for layer.merge
                    if ForwardGraph._is_recurrent_layer(layer, input_layer, architecture):
                        if input_layer not in recurrent_layers:
                            variable_input = G.Variable(*architecture._layers[input_layer].layer.output_shape)
                            recurrent_layers[input_layer] = variable_input
                        else:
                            variable_input = recurrent_layers[input_layer]
                        input_layers.append(variable_input)
                    else:
                        input_layers.append(self.layers[input_layer])
                self.layers[layer.name] = layers[layer.name](
                    layer.merge(
                        self.layers[layer.merge.main_input],
                        *input_layers,
                    ),
                )
            # On processing layer, check recurrent marks
            # if so, assign layer value to this variable and unmark layer
            # use assign result as layer in graph
            if layer.name in recurrent_layers:
                self.layers[layer.name] = recurrent_layers[layer.name].assign(self.layers[layer.name])
                del recurrent_layers[layer.name]

        self.outputs = tuple([self.layers[x] for x in architecture._output_layers])
        self.session.run(G.global_variables_initializer())

    def _get_parameters_vector(self, architecture: Architecture) -> G.VariableType:
        parameters_flatten: List[np.ndarray] = []
        layer: TrainableLayer
        index = 0
        for layer in architecture.trainable:
            self._parameters_index[layer.name] = index
            parameters_flatten.extend((layer.weights.flatten(), layer.biases.flatten()))
            index += np.prod(parameters_flatten[-2].shape) + np.prod(  # type: ignore[call-overload]
                parameters_flatten[-1].shape
            )
        return G.Variable(np.hstack(parameters_flatten))

    def _get_parameters_from_vector(self, layer: TrainableLayer) -> Tuple[G.Tensor, G.Tensor]:
        layer_index = self._parameters_index[layer.name]
        weights = G.reshape(
            self.parameters[layer_index : layer_index + layer.weights_shape.prod], layer.weights_shape.dimension
        )
        biases = G.reshape(
            self.parameters[
                layer_index
                + layer.weights_shape.prod : layer_index
                + layer.weights_shape.prod
                + layer.biases_shape.prod
            ],
            layer.biases_shape.dimension,
        )
        return weights, biases

    # pylint:disable=protected-access
    @staticmethod
    def _is_recurrent_layer(layer: Layer, input_layer: str, architecture: Architecture) -> bool:
        return architecture._layers[layer.name].layer_id < architecture._layers[input_layer].layer_id

    # pylint:enable=protected-access

    def _create_layers(self, batch_size: int, architecture: Architecture) -> Dict[str, Callable[..., G.Tensor]]:
        layers: Dict[str, Callable[..., G.Tensor]] = {}
        for layer in architecture.layers:
            weights = None
            biases = None
            if isinstance(layer, TrainableLayer):
                weights, biases = self._get_parameters_from_vector(layer)
            layers[layer.name] = LayerFunc.get_layer_func(layer, batch_size=batch_size, weights=weights, biases=biases)
        return layers

    def describe(self) -> None:
        ...

    # FIXME Get List[np.nparray] for x_data, we can use more than 1 input layers

    def predict_proba(self, x_data: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        dataset = self._prepare_batch(x_data, shuffle=False)
        predict_batches, _ = self.session.run(self._get_batches(dataset, train=False))
        predict_result = list(self.session.run(self.outputs, feed_dict={self.inputs: predict_batches[0]}))
        for batch in predict_batches[1:]:
            for pred_idx, (pred_mem, pred_current) in enumerate(
                zip(predict_result, self.session.run(self.outputs, feed_dict={self.inputs: batch}))
            ):
                predict_result[pred_idx] = np.vstack([pred_mem, pred_current])
        return predict_result[0] if len(predict_result) == 1 else predict_result

    def predict(
        self, x_data: np.ndarray, thresholds: Optional[Union[float, List[float]]] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get class labels from predict_proba() with list threshold for all output layers.
        default threshold: None - 0.5 for all output layers.
        """

        predict_results = self.predict_proba(x_data)
        if isinstance(predict_results, np.ndarray):
            predict_results = [predict_results]
        if thresholds is None:
            _thresholds = [0.5 for idx in range(len(predict_results))]
        else:
            if isinstance(thresholds, float):
                _thresholds = [thresholds]
            else:
                _thresholds = thresholds
        for out_idx, (predict_result, threshold) in enumerate(zip(predict_results, _thresholds)):
            predict_results[out_idx] = (predict_result > threshold).astype(int)
        return predict_results[0] if len(predict_results) == 1 else predict_results
