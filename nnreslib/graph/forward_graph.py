from __future__ import annotations

import logging
from typing import Callable, Dict, List, NamedTuple, Union

from .layers import LayerFunc
from ..backend import graph as G
from ..layers import Layer, TrainableLayer
from ..model import Model

_logger = logging.getLogger(__name__)

# pylint:disable=protected-access
Parameters = NamedTuple("Parameters", [("weights", G._variable), ("biases", G._variable)])
# pylint:enable=protected-access

# FIXME: Global fix G(TF) type annotations


class ForwardGraph:
    __slots__ = (
        # "batch_size",
        # "model",
        # "input_data",
        "input_labels",
        "layers",
        "parameters",
        # "output",
        # "vector_error",
        # "train_loss",
    )

    def __init__(self, batch_size: int, model: Model) -> None:
        # self.batch_size = batch_size
        # self.model = model

        # XXX: This exist in self.layers. It's really need?
        # self.input_data = {
        #     x.name: G.placeholder(name=x.name, shape=(batch_size, *x.output_shape)) for x in model.input_layers
        # }

        # XXX: Is it calculated on output layer's shapes??
        self.input_labels = [
            G.placeholder(name=x.name, shape=(batch_size, *x.output_shape)) for x in model.output_layers
        ]  # TODO: may be output_data???

        self.parameters: List[Parameters] = []

        _logger.debug("Start building model graph")  # TODO: debug or info?
        layers = self._create_layers(batch_size, model)

        # Move placeholders for input layers to graph layers dict
        self.layers: Dict[str, Union[Callable[..., G.Tensor], G.Tensor]] = {
            x: layers[x] for x in model._input_layers
        }  # TODO: layer is Callable or Tensor?

        # dict value is G.variable
        recurrent_layers: Dict[str, G.Tensor] = {}

        # layers: Dict with partial defined layers: LayerFunc for layers, and placeholder for InputLayers.
        # Parameter for LayerFunc - input
        # self.layers: Dict with initialized layers: input is passed, really created graph's nodes
        layer: Layer
        for layer, inputs in model.architecture:  # BUG: check architecture for InputLayer in middle of Model definition
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
                    if ForwardGraph._is_recurrent_layer(layer, input_layer, model):
                        if input_layer not in recurrent_layers:
                            variable_input = G.variable(*model._layers[input_layer].layer.output_shape)
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

        # # tf.Varibale([w+b+w1+b1])

        # self.output = tf.squeeze(current_node)
        # self.vector_error = self.input_labels - self.settings.outputs
        # self.train_loss = tf.reduce_mean(tf.square(self.vector_error), name="Train_loss")

    # pylint:disable=protected-access
    @staticmethod
    def _is_recurrent_layer(layer: Layer, input_layer: str, model: Model) -> bool:
        return model._layers[layer.name].layer_id < model._layers[input_layer].layer_id

    # pylint:enable=protected-access

    def _create_layers(self, batch_size: int, model: Model) -> Dict[str, Callable[..., G.Tensor]]:
        layers: Dict[str, Callable[..., G.Tensor]] = {}
        for layer in model.layers:
            if isinstance(layer, TrainableLayer):
                weights = G.variable(layer.weights)
                biases = G.variable(layer.biases)
                self.parameters.append(Parameters(weights, biases))
            layers[layer.name] = LayerFunc.get_layer_func(layer, batch_size=batch_size, weights=weights, biases=biases)
        return layers

    def describe(self) -> None:
        ...
