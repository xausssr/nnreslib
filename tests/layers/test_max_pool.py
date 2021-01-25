from nnreslib.layers import MaxPoolLayer
from nnreslib.utils.types import Shape


def test_serialize():
    assert MaxPoolLayer("mp1", Shape(1, 2), Shape(5, 6)).serialize() == dict(
        name="mp1", type="MaxPoolLayer", merge=None, is_out=False, kernel=[1, 2], stride=[5, 6]
    )
