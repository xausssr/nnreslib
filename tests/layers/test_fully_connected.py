import pytest

from nnreslib.layers import FullyConnectedLayer
from nnreslib.utils.types import Shape


def test_neurons_validation():
    with pytest.raises(ValueError, match=r"'neurons' must be greater than 0"):
        FullyConnectedLayer("l1", 0)
    with pytest.raises(ValueError, match=r"'neurons' must be greater than 0"):
        FullyConnectedLayer("l1", -1)


def test_set_shapes():
    layer = FullyConnectedLayer("fc_1", 10)
    with pytest.raises(ValueError, match=r"Incorrect input shape.*fc_1"):
        layer.set_shapes(Shape(2, 2))
    with pytest.raises(ValueError, match=r"Incorrect input shape.*fc_1"):
        layer.set_shapes(Shape())


def test_serialize():
    assert FullyConnectedLayer("fc_1", 5).serialize() == dict(
        name="fc_1",
        type="FullyConnectedLayer",
        merge=None,
        is_out=False,
        activation="SIGMOID",
        initializer=dict(weights_initializer="HE_NORMAL", biases_initializer="ZEROS"),
        neurons=5,
    )

    assert FullyConnectedLayer("fc_1", 4).serialize() == dict(
        name="fc_1",
        type="FullyConnectedLayer",
        merge=None,
        is_out=False,
        activation="SIGMOID",
        initializer=dict(weights_initializer="HE_NORMAL", biases_initializer="ZEROS"),
        neurons=4,
    )
