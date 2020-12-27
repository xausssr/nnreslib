from __future__ import annotations

from typing import Any

# from typing import List, Tuple

# from .layers import ConvolutionLayer, FlattenLayer, FullyConnectedLayer, MaxPoolLayer
# from .utils.tf_helper import tf


class ForwardGraph:
    __slots__ = (
        "batch_size",
        "model",
        "input_data",
        "input_labels",
        "parameters",
        "output",
        "vector_error",
        "train_loss",
    )

    # TODO: what about dtype???

    # TODO: REWRITE!!!!!

    def __init__(self, batch_size: int, model: Any) -> None:
        self.batch_size = batch_size
        self.model = model

        # self.input_data = tf.Variable(
        #     name="input_data", dtype=tf.float64, shape=(self.settings.batch_size, *self.settings.inputs)
        # )
        # self.input_labels = tf.Variable(
        #     name="input_labels", dtype=tf.float64, shape=(self.settings.batch_size, *self.settings.outputs)
        # )  # TODO: check name

        # self.parameters: List[Tuple[tf.Variable, tf.Variable]] = []

        # # Start building. TODO: may be need log it
        # current_node = self.input_data
        # for layer in self.settings.model.layers:
        #     if isinstance(layer, ConvolutionLayer):
        #         weights = tf.Variable(layer.weights)
        #         biases = tf.Variable(layer.biases)
        #         self.parameters.append((weights, biases))
        #         pad = layer.pad / 2
        #         current_node = layer.activation(
        #             tf.nn.conv2d(
        #                 current_node,
        #                 weights,
        #                 (1, *layer.stride, 1),  # TODO: feature request
        #                 padding=(0, 0, *pad, 0, 0),  # TODO: feature request
        #             )
        #             + biases
        #         )
        #     elif isinstance(layer, MaxPoolLayer):
        #         current_node = tf.nn.max_pool(
        #             current_node,
        #             (1, *layer.kernel, 1),
        #             (1, *layer.stride, 1),
        #             padding="SAME",  # TODO: need fix???
        #         )
        #     elif isinstance(layer, FlattenLayer):
        #         current_node = tf.reshape(current_node, (self.settings.batch_size, *layer.output_shape))
        #     elif isinstance(layer, FullyConnectedLayer):
        #         weights = tf.Variable(layer.weights)
        #         biases = tf.Variable(layer.biases)
        #         self.parameters.append((weights, biases))
        #         current_node = layer.activation(tf.matmul(current_node, weights) + biases)

        # # tf.Varibale([w+b+w1+b1])

        # self.output = tf.squeeze(current_node)
        # self.vector_error = self.input_labels - self.settings.outputs
        # self.train_loss = tf.reduce_mean(tf.square(self.vector_error), name="Train_loss")

    def describe(self) -> None:
        ...
