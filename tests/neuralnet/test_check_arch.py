from typing import List

from nnreslib.neuralnet import NeuralNet
from nnreslib.settings import Settings
from nnreslib.utils.layers import ConvolutionLayer, FlattenLayer, FullyConnectedLayer, Layer, MaxPoolLayer
from nnreslib.utils.types import ActivationFunction, Shape

base_architecture: List[Layer] = [
    ConvolutionLayer(
        "convolution1", Shape(3, 3), Shape(2, 2), 15, Shape(0, 0, is_null=True), ActivationFunction.SIGMOID
    ),
    ConvolutionLayer(
        "convolution2", Shape(3, 3), Shape(1, 1), 16, Shape(0, 0, is_null=True), ActivationFunction.SIGMOID
    ),
    MaxPoolLayer("pooling1", Shape(3, 3), Shape(1, 1)),
    FlattenLayer("flatten"),
    FullyConnectedLayer("fullyconnect1", 13, ActivationFunction.SIGMOID),
    MaxPoolLayer("pooling2", Shape(3, 3), Shape(1, 1)),
]

base_settings = Settings(
    1,
    2,
    Shape(
        250,
        250,
        3,
    ),
    base_architecture,
)


def test_max_pool_after_fully_connected():

    neural_network = NeuralNet(base_settings)
    assert not neural_network.check_arch()


def test_conv_after_fully_connected():
    architecture = base_architecture[:-1]
    settings = base_settings
    settings.architecture = architecture
    architecture.append(
        ConvolutionLayer("convolution3", Shape(3, 3), Shape(1, 1), 7, Shape(1, 1), ActivationFunction.SIGMOID)
    )
    neural_network = NeuralNet(settings)
    assert not neural_network.check_arch()


def test_max_pool_after_fully_flatten():
    architecture = base_architecture[:-2]
    architecture.append(MaxPoolLayer("pooling2", Shape(3, 3), Shape(1, 1)))
    settings = base_settings
    settings.architecture = architecture
    neural_network = NeuralNet(settings)
    assert not neural_network.check_arch()


def test_conv_after_fully_flatten():
    architecture = base_architecture[:-2]
    architecture.append(
        ConvolutionLayer("convolution3", Shape(3, 3), Shape(1, 1), 28, Shape(1, 1), ActivationFunction.SIGMOID)
    )
    settings = base_settings
    settings.architecture = architecture
    neural_network = NeuralNet(settings)
    assert not neural_network.check_arch()


def test_normal_behavior():
    architecture = base_architecture[:-1]
    settings = base_settings
    settings.architecture = architecture
    neural_network = NeuralNet(settings)
    assert neural_network.check_arch()


def test_zero_conv_result():
    architecture = base_architecture[:-1]
    settings = Settings(
        1,
        2,
        Shape(
            7,
            7,
            3,
        ),
        architecture,
    )
    neural_network = NeuralNet(settings)
    assert not neural_network.check_arch()


def test_zero_arch():
    settings = Settings(
        1,
        2,
        Shape(
            2,
            2,
            3,
        ),
        [],
    )
    neural_network = NeuralNet(settings)
    assert not neural_network.check_arch()
