import pytest

from nnreslib.settings import Settings
from nnreslib.utils.layers import FlattenLayer
from nnreslib.utils.types import Shape


def test_settings_init():
    with pytest.raises(TypeError, match=r"missing 4 required"):
        Settings()
    with pytest.raises(TypeError, match=r"missing 3 required"):
        Settings(1)
    with pytest.raises(TypeError, match=r"missing 2 required"):
        Settings(1, 10)
    with pytest.raises(TypeError, match=r"missing 1 required"):
        Settings(1, 10, Shape(10, 10, 1))
    assert Settings(1, 10, Shape(10, 10, 1), [FlattenLayer("test", 10)])
