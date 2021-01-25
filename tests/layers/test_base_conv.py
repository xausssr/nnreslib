import pytest

from nnreslib.layers import BaseConvLayer
from nnreslib.utils.types import Shape


class SerializableBaseConvLayer(BaseConvLayer):
    def _set_output_shape(self) -> None:
        self._output_shape = None


def test_base_param_validation():
    with pytest.raises(ValueError, match=r"kernel.*0"):
        SerializableBaseConvLayer("bad_kernel", Shape(), Shape(1), 10)
    with pytest.raises(ValueError, match=r"stride.*0"):
        SerializableBaseConvLayer("bad_stride", Shape(1), Shape(), 10)


def test_serialize():
    assert SerializableBaseConvLayer("base_1", Shape(1, 2), Shape(5, 6)).serialize() == dict(
        name="base_1", type="SerializableBaseConvLayer", merge=None, is_out=False, kernel=[1, 2], stride=[5, 6]
    )
    assert SerializableBaseConvLayer("base_1", Shape(1, 2), Shape(0, 0, is_null=True)).serialize() == dict(
        name="base_1",
        type="SerializableBaseConvLayer",
        merge=None,
        is_out=False,
        kernel=[1, 2],
        stride=dict(shape=[0, 0], is_null=True),
    )
