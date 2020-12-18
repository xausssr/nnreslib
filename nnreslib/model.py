from __future__ import annotations

from typing import Tuple

from .layers import ConvolutionLayer, FlattenLayer, FullyConnectedLayer, InputLayer, Layer, MaxPoolLayer, TrainableLayer


class Model:
    __slots__ = ("input_layer", "layers", "neurons_count")

    def __init__(self, input_layer: InputLayer, *layers: Layer):
        self.input_layer = input_layer
        self.layers: Tuple[Layer, ...] = layers
        self.neurons_count = 0
        self._build_model()

    def _build_model(self) -> None:
        pre_layer: Layer = self.input_layer
        last_filters_count = pre_layer.get_last_filters_count()

        for layer in self.layers:
            if not Model._check_layers_type(pre_layer, layer):
                raise ValueError(f"Unsupported layer sequence: {pre_layer.name} -> {layer.name}")

            last_filters_count = layer.get_last_filters_count(last_filters_count)
            layer.input_shape = pre_layer.output_shape
            layer.set_output_shape(last_filters_count=last_filters_count)
            self._init_weights(layer)

            self.neurons_count += layer.neurons_count
            pre_layer = layer

    @staticmethod
    def _check_layers_type(pre_layer: Layer, layer: Layer) -> bool:
        return (
            isinstance(pre_layer, InputLayer)
            or isinstance(pre_layer, (ConvolutionLayer, MaxPoolLayer))
            and isinstance(layer, (ConvolutionLayer, MaxPoolLayer, FlattenLayer))
            or isinstance(pre_layer, (FullyConnectedLayer, FlattenLayer))
            and isinstance(layer, FullyConnectedLayer)
        )

    def _init_weights(self, layer: Layer) -> None:
        if isinstance(layer, TrainableLayer):
            layer.set_weights(self.input_layer.input_mean, self.input_layer.input_std)
            layer.set_biases(self.input_layer.input_mean, self.input_layer.input_std)
