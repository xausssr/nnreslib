from nnreslib.layers import InputLayer
from nnreslib.utils.types import Shape


def test_layer_shapes():
    input_layer = InputLayer("input", Shape(250, 250, 3))
    assert input_layer.output_shape == (250, 250, 3)
    input_layer.set_shapes(Shape())
    assert input_layer.output_shape == (250, 250, 3)
    input_layer._set_output_shape()  # pylint:disable=protected-access
    assert input_layer.output_shape == (250, 250, 3)


def test_serialize():
    assert InputLayer("input", Shape(250, 250, 3)).serialize() == dict(
        name="input", type="InputLayer", input_shape=[250, 250, 3]
    )
    assert InputLayer("input", Shape(1, 2)).serialize() == dict(name="input", type="InputLayer", input_shape=[1, 2])
