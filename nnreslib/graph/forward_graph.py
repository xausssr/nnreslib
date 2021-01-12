from __future__ import annotations

import logging
from typing import Callable, Dict, List, NamedTuple, Union

from .layers import LayerFunc
from ..architecture import Architecture
from ..backend import graph as G
from ..layers import Layer, TrainableLayer

_logger = logging.getLogger(__name__)

Parameters = NamedTuple("Parameters", [("weights", G.VariableType), ("biases", G.VariableType)])

# FIXME: Global fix G(TF) type annotations


class ForwardGraph:
    __slots__ = (
        # "batch_size",
        "layers",
        "parameters",
        "output",
        # "vector_error",
        # "train_loss",
    )

    def __init__(self, batch_size: int, architecture: Architecture) -> None:
        # self.batch_size = batch_size

        self.parameters: List[Parameters] = []

        _logger.info("Start building model graph")
        layers = self._create_layers(batch_size, architecture)

        # Move placeholders for input layers to graph layers dict
        self.layers: Dict[str, Union[Callable[..., G.Tensor], G.Tensor]] = {
            x: layers[x] for x in architecture._input_layers
        }  # FIXME: layer is Callable[..., G.Tensor]

        # dict value is G.variable
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
                    # if so, create G.variable(with layer shape and 0 init value) and mark recurrent layer
                    # Use this variable as argument for layer.merge
                    if ForwardGraph._is_recurrent_layer(layer, input_layer, architecture):
                        if input_layer not in recurrent_layers:
                            variable_input = G.variable(*architecture._layers[input_layer].layer.output_shape)
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

    # pylint:disable=protected-access
    @staticmethod
    def _is_recurrent_layer(layer: Layer, input_layer: str, architecture: Architecture) -> bool:
        return architecture._layers[layer.name].layer_id < architecture._layers[input_layer].layer_id

    # pylint:enable=protected-access

    def _create_layers(self, batch_size: int, architecture: Architecture) -> Dict[str, Callable[..., G.Tensor]]:
        layers: Dict[str, Callable[..., G.Tensor]] = {}
        for layer in architecture.layers:
            if isinstance(layer, TrainableLayer):
                weights = G.variable(layer.weights)
                biases = G.variable(layer.biases)
                self.parameters.append(Parameters(weights, biases))
            layers[layer.name] = LayerFunc.get_layer_func(layer, batch_size=batch_size, weights=weights, biases=biases)
        return layers

    def describe(self) -> None:
        ...
