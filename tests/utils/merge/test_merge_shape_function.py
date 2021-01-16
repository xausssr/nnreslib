import pytest

from nnreslib.utils.merge import _check_merged_tensors, _merge_flatten, _merge_to_main
from nnreslib.utils.types import Shape


def test_check_merged_tensor_wrong_shapes():
    with pytest.raises(ValueError, match="Input tensors have a different number of dimentions"):
        _check_merged_tensors(
            Shape(12, 32, 12),
            Shape(12, 32),
            Shape(32, 12),
        )
    with pytest.raises(ValueError, match="Input tensors have a different number of dimentions"):
        _check_merged_tensors(
            Shape(12, 32, 12),
            Shape(12, 12, 32),
            Shape(12, 32, 12),
            recurrent=[Shape(31, 12), Shape(21, 12, 12)],
        )


def test_check_merged_tensor_correct_shapes():
    _check_merged_tensors(Shape(12, 32, 12))
    _check_merged_tensors(
        Shape(12, 32, 12),
        Shape(12, 32, 12),
        Shape(32, 12, 12),
    )
    _check_merged_tensors(
        Shape(12, 32, 12),
        Shape(12, 12, 32),
        Shape(12, 32, 12),
        recurrent=[Shape(31, 12, 12), Shape(21, 12, 12)],
    )


def test_merge_flatten():
    assert _merge_flatten(Shape(20)) == Shape(20)
    assert (
        _merge_flatten(
            Shape(14),
            Shape(12),
            Shape(17),
            Shape(20),
        )
        == Shape(63)
    )
    assert (
        _merge_flatten(
            Shape(28),
            Shape(18),
            recurrent=[
                Shape(18),
                Shape(20),
            ],
        )
        == Shape(84)
    )


def test_merge_to_main():
    assert (
        _merge_to_main(
            Shape(10),
            Shape(11),
            Shape(12),
        )
        == Shape(33)
    )
    assert (
        _merge_to_main(
            Shape(28),
            Shape(18),
            recurrent=[
                Shape(18),
                Shape(20),
            ],
        )
        == Shape(84)
    )
    assert (
        _merge_to_main(
            Shape(18, 23, 12, 12, 8),
            Shape(28, 21, 14, 14, 3),
            Shape(18, 18, 17, 17, 9),
            Shape(13, 13, 20, 21, 7),
        )
        == Shape(18, 23, 12, 12, 27)
    )
    assert (
        _merge_to_main(
            Shape(13, 13, 20, 21, 7),
            Shape(28, 21, 14, 14, 3),
            Shape(18, 18, 17, 17, 9),
            recurrent=(Shape(13, 13, 20, 21, 7),),
        )
        == Shape(13, 13, 20, 21, 26)
    )
