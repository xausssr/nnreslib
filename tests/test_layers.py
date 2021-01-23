import pytest

from nnreslib.layers import ConvolutionLayer, FlattenLayer, FullyConnectedLayer, InputLayer
from nnreslib.utils.types import Shape


def test_conv_exceptions():
    with pytest.raises(ValueError, match=r"kernel.*0"):
        ConvolutionLayer("conv_bad", Shape(), Shape(1), 10)
    with pytest.raises(ValueError, match=r"stride.*0"):
        ConvolutionLayer("conv_bad", Shape(1), Shape(), 10)
    with pytest.raises(ValueError, match=r"filters.*0"):
        ConvolutionLayer("conv_bad", Shape(1), Shape(1), 0)


def test_fully_connected_exceptions():
    with pytest.raises(ValueError, match=r"neurons.*0"):
        FullyConnectedLayer("fc_bad", 0)

    layer = FullyConnectedLayer("fc_1", 10)
    with pytest.raises(ValueError, match=r"Incorrect input shape.*fc_1"):
        layer.set_shapes(Shape(2, 2))
    with pytest.raises(ValueError, match=r"Incorrect input shape.*fc_1"):
        layer.set_shapes(Shape())


def test_input_layer():
    input_layer = InputLayer("input", Shape(250, 250, 3))
    assert input_layer.output_shape == (250, 250, 3)
    input_layer.set_shapes(Shape())
    assert input_layer.output_shape == (250, 250, 3)
    input_layer._set_output_shape()  # pylint:disable=protected-access
    assert input_layer.output_shape == (250, 250, 3)


def test_layer_shapes_exceptions():
    layer = FlattenLayer("test")
    with pytest.raises(ValueError, match="set_shapes"):
        layer.input_shape  # pylint:disable=pointless-statement
    with pytest.raises(ValueError, match="set_shapes"):
        layer.output_shape  # pylint:disable=pointless-statement
