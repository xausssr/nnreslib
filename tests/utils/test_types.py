import pytest

from nnreslib.utils.types import Shape


def test_shape_from_list():
    with pytest.raises(ValueError, match=r".*int$"):
        Shape([1, 2, 3])


def test_shape_from_tuple():
    with pytest.raises(ValueError, match=r".*int$"):
        Shape((1, 2, 3))


def test_shape_with_zero():
    with pytest.raises(ValueError, match=r".*negative*"):
        Shape(1, 0, 3)


def test_shape_with_negative_dimension():
    with pytest.raises(ValueError, match=r".*negative*"):
        Shape(2, -3)


def test_shape_init():
    assert Shape().dimension == ()
    assert Shape(1).dimension == (1,)
    assert Shape(10, 20).dimension == (10, 20)
    assert Shape(1, 2, 3, 4, 5).dimension == (1, 2, 3, 4, 5)


def test_shape_mul():
    assert Shape() == 2 * Shape()
    assert Shape() == Shape() * 2
    assert Shape(2, 4) == Shape(1, 2) * 2
    assert Shape(2, 4) == 2 * Shape(1, 2)
    assert Shape(4, 8) == 2 * Shape(1, 2) * 2


def test_shape_prod():
    assert Shape().prod == 0
    assert Shape(1, 2, 3).prod == 6
    assert Shape(1).prod == 1


def test_shape_iter():
    assert list(Shape()) == []
    assert list(Shape(1, 2, 3)) == [1, 2, 3]


def test_shape_index():
    with pytest.raises(IndexError):
        Shape()[0]  # pylint:disable=expression-not-assigned
    with pytest.raises(IndexError):
        Shape(1, 2, 3)[10]  # pylint:disable=expression-not-assigned
    assert Shape(1, 2, 3)[1] == 2
