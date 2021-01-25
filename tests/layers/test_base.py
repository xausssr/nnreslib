import pytest

from nnreslib.layers import Layer
from nnreslib.utils.merge import MergeInputs
from nnreslib.utils.types import Shape


class SerializableBaseLayer(Layer):
    def _set_output_shape(self) -> None:
        self._output_shape = None


def test_get_shapes():
    layer = SerializableBaseLayer("base_1")
    with pytest.raises(ValueError, match="set_shapes"):
        layer.input_shape  # pylint:disable=pointless-statement
    with pytest.raises(ValueError, match="set_shapes"):
        layer.output_shape  # pylint:disable=pointless-statement


def test_check_input_shape():
    # pylint:disable=protected-access
    assert SerializableBaseLayer("b1")._check_input_shape(Shape(0, is_null=True))
    assert SerializableBaseLayer("b1")._check_input_shape(Shape(1, 2, 3))
    # pylint:enable=protected-access


def test_set_shapes():
    SerializableBaseLayer("b1").set_shapes(Shape(1, 2))


def test_neurons_count():
    assert SerializableBaseLayer("b1").neurons_count == 0


def test_serialize():
    assert SerializableBaseLayer("base_1").serialize() == dict(
        name="base_1", type="SerializableBaseLayer", merge=None, is_out=False
    )
    assert SerializableBaseLayer("base_1", is_out=True).serialize() == dict(
        name="base_1", type="SerializableBaseLayer", merge=None, is_out=True
    )
    assert SerializableBaseLayer("base_1", merge=MergeInputs()).serialize() == dict(
        name="base_1",
        type="SerializableBaseLayer",
        merge=dict(main_input="", merge_func="RESHAPE_TO_MAIN"),
        is_out=False,
    )
    assert SerializableBaseLayer("base_1", merge=MergeInputs(), is_out=True).serialize() == dict(
        name="base_1",
        type="SerializableBaseLayer",
        merge=dict(main_input="", merge_func="RESHAPE_TO_MAIN"),
        is_out=True,
    )
