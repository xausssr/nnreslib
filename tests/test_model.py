from nnreslib.architecture import ArchitectureType
from nnreslib.layers import FullyConnectedLayer, InputLayer
from nnreslib.model import Model
from nnreslib.utils.types import Shape


def test_serialize():
    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        FullyConnectedLayer("fc_2", neurons=6),
        FullyConnectedLayer("fc_3", neurons=3, is_out=True),
    ]

    model = Model(150, architecture)

    simple_serialization = model.serialize()
    assert simple_serialization["batch_size"] == 150
    assert not simple_serialization["architecture"]["is_built"]
    assert simple_serialization["architecture"]["architecture"] == model.architecture.serialize()

    built_serialization = model.serialize(True)
    assert built_serialization["batch_size"] == 150
    assert built_serialization["architecture"]["is_built"]
    assert built_serialization["architecture"]["architecture"] == model.architecture.serialize(True)
