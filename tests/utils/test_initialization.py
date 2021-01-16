from nnreslib.utils.initialization import Initialization, StandartInitializer


def test_serialize():
    assert Initialization().serialize() == dict(weights_initializer="HE_NORMAL", biases_initializer="ZEROS")
    assert Initialization(StandartInitializer.HAYKIN).serialize() == dict(
        weights_initializer="HAYKIN", biases_initializer="ZEROS"
    )
    assert Initialization(StandartInitializer.ZEROS, StandartInitializer.ZEROS).serialize() == dict(
        weights_initializer="ZEROS", biases_initializer="ZEROS"
    )
    assert Initialization(biases_initializer=StandartInitializer.HAYKIN).serialize() == dict(
        weights_initializer="HE_NORMAL", biases_initializer="HAYKIN"
    )
    assert Initialization(weights_initializer=lambda x: x + 1).serialize() == dict(
        weights_initializer=dict(function="custom"), biases_initializer="ZEROS"
    )
    assert Initialization(lambda x: x + 1, lambda x: x - 1).serialize() == dict(
        weights_initializer=dict(function="custom"), biases_initializer=dict(function="custom")
    )
