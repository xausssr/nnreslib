import pytest

from nnreslib.layers import TrainableLayer
from nnreslib.utils.initialization import Initialization, StandartInitializer
from nnreslib.utils.types import ActivationFunctions, Shape


class SerializableTrainableLayer(TrainableLayer):
    def _set_output_shape(self) -> None:
        self._output_shape = None

    @property
    def weights_shape(self) -> Shape:
        return Shape(1)

    @property
    def biases_shape(self) -> Shape:
        return Shape(1)


def test_parameters_not_initialized():
    layer = SerializableTrainableLayer("train_1", ActivationFunctions.RELU)
    with pytest.raises(ValueError, match=r"Weights.*not initialized"):
        layer.weights  # pylint:disable=pointless-statement
    with pytest.raises(ValueError, match=r"Biases.*not initialized"):
        layer.biases  # pylint:disable=pointless-statement


def test_serialize():
    assert SerializableTrainableLayer("train_1", ActivationFunctions.RELU).serialize() == dict(
        name="train_1",
        type="SerializableTrainableLayer",
        merge=None,
        is_out=False,
        activation="RELU",
        initializer=dict(weights_initializer="HE_NORMAL", biases_initializer="ZEROS"),
    )

    assert SerializableTrainableLayer(
        "train_1", ActivationFunctions.SIGMOID, initializer=Initialization(StandartInitializer.HAYKIN)
    ).serialize() == dict(
        name="train_1",
        type="SerializableTrainableLayer",
        merge=None,
        is_out=False,
        activation="SIGMOID",
        initializer=dict(weights_initializer="HAYKIN", biases_initializer="ZEROS"),
    )
