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

    # XXX: support serialize CustomInitializer
    assert Initialization(weights_initializer=lambda x: x + 1).serialize() == dict(
        weights_initializer=dict(function="custom"), biases_initializer="ZEROS"
    )
    assert Initialization(lambda x: x + 1, lambda x: x - 1).serialize() == dict(
        weights_initializer=dict(function="custom"), biases_initializer=dict(function="custom")
    )


def test_deserialize():
    initialization = Initialization()
    assert Initialization.deserialize(initialization.serialize()) == initialization

    initialization = Initialization(StandartInitializer.HAYKIN)
    assert Initialization.deserialize(initialization.serialize()) == initialization

    initialization = Initialization(StandartInitializer.ZEROS, StandartInitializer.ZEROS)
    assert Initialization.deserialize(initialization.serialize()) == initialization

    initialization = Initialization(biases_initializer=StandartInitializer.HAYKIN)
    assert Initialization.deserialize(initialization.serialize()) == initialization

    # XXX: support deserialize CustomInitializer
    initialization = Initialization(weights_initializer=lambda x: x + 1)
    deserialized_initialization = Initialization.deserialize(initialization.serialize())
    assert deserialized_initialization != initialization
    assert deserialized_initialization.weights(5) == 5
    assert deserialized_initialization.biases == initialization.biases

    initialization = Initialization(lambda x: x + 1, lambda x: x - 1)
    deserialized_initialization = Initialization.deserialize(initialization.serialize())
    assert deserialized_initialization != initialization
    assert deserialized_initialization.weights(5) == 5
    assert deserialized_initialization.biases(10) == 10
