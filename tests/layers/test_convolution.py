import pytest

from nnreslib.layers import ConvolutionLayer
from nnreslib.utils.types import Shape


def test_filter_validation():
    with pytest.raises(ValueError, match=r"'filters' must be greater than 0"):
        ConvolutionLayer("l1", Shape(1, 2), Shape(5, 6), 0)
    with pytest.raises(ValueError, match=r"'filters' must be greater than 0"):
        ConvolutionLayer("l1", Shape(1, 2), Shape(5, 6), -1)


def test_serialize():
    assert ConvolutionLayer("l1", Shape(1, 2), Shape(5, 6), 10).serialize() == dict(
        name="l1",
        type="ConvolutionLayer",
        merge=None,
        is_out=False,
        kernel=[1, 2],
        stride=[5, 6],
        activation="RELU",
        initializer=dict(weights_initializer="HE_NORMAL", biases_initializer="ZEROS"),
        filters=10,
        pad=dict(shape=[0, 0], is_null=True),
    )

    assert ConvolutionLayer("l1", Shape(1, 2), Shape(5, 6), 15, Shape(1, 1)).serialize() == dict(
        name="l1",
        type="ConvolutionLayer",
        merge=None,
        is_out=False,
        kernel=[1, 2],
        stride=[5, 6],
        activation="RELU",
        initializer=dict(weights_initializer="HE_NORMAL", biases_initializer="ZEROS"),
        filters=15,
        pad=[1, 1],
    )
