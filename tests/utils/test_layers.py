import pytest

from nnreslib.utils.layers import ConvolutionLayer, FlattenLayer, FullyConnectedLayer
from nnreslib.utils.types import ActivationFunction, Shape


def test_conv_make_out_shape():
    layer = ConvolutionLayer("conv_1", Shape(3, 3), Shape(2, 2), 15, Shape(0, 0, is_null=True), ActivationFunction.RELU)
    assert layer.make_out_shape(Shape(1, 2), Shape(1, 1)) == Shape(1, 2)
    assert layer.make_out_shape(Shape(250, 250, 3), Shape(0, 0, is_null=True)) == Shape(125, 125)

    with pytest.raises(ValueError, match="negative"):
        layer.make_out_shape(Shape(1, 2), Shape(0, 0, is_null=True))


def test_flatten_make_out_shape():
    layer = FlattenLayer("flat_1")
    assert layer.make_out_shape(Shape(1, 2), Shape(0, 0, is_null=True)) == Shape(2)
    assert layer.make_out_shape(Shape(10, 20, 30), Shape(0, 0, is_null=True)) == Shape(6000)


def test_fully_connected_make_out_shape():
    layer = FullyConnectedLayer("fully_con_1", 1, ActivationFunction.RELU)
    assert layer.make_out_shape(Shape(0, 0, is_null=True), Shape(0, 0, is_null=True)) == Shape()
