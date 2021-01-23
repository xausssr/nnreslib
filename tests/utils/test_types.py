import pytest

from nnreslib.utils.types import Shape


def test_shape_from_none():
    with pytest.raises(ValueError, match=r".*int$"):
        Shape((1, None))

    with pytest.raises(ValueError, match=r".*int$"):
        Shape([None])


def test_shape_with_zero():
    with pytest.raises(ValueError, match=r".*negative.*"):
        Shape(1, 0, 3)


def test_shape_with_negative_dimension():
    with pytest.raises(ValueError, match=r".*negative.*"):
        Shape(2, -3)


def test_shape_from_list():
    assert Shape([]) == ()
    assert Shape([1]) == (1,)
    assert Shape([1, 2, 3]) == (1, 2, 3)
    assert Shape([0, 0, 0], is_null=True) == (0, 0, 0)


def test_shape_from_tuple():
    assert Shape(()) == ()
    assert Shape((1,)) == (1,)
    assert Shape((1, 2, 3)) == (1, 2, 3)
    assert Shape((0, 0, 0), is_null=True) == (0, 0, 0)


def test_shape_init():
    assert Shape(None) == ()
    assert Shape() == ()
    assert Shape(1) == (1,)
    assert Shape(10, 20) == (10, 20)
    assert Shape(1, 2, 3, 4, 5) == (1, 2, 3, 4, 5)
    assert Shape(1, 0, is_null=True) == (1, 0)
    assert Shape(0, 0, 0, is_null=True) == (0, 0, 0)


def test_shape_mul():
    assert 2 * Shape() == Shape()
    assert Shape() * 2 == Shape()
    assert Shape(0, is_null=True) * 10 == (0,)
    assert Shape(0, 0, 0, is_null=True) * 10 == (0, 0, 0)
    assert Shape(1, 2) * 2 == Shape(2, 4)
    assert 2 * Shape(1, 2) == Shape(2, 4)
    assert 2 * Shape(1, 2) * 2 == Shape(4, 8)
    with pytest.raises(TypeError, match=r".*unsupported.*str"):
        Shape(4, 2) * "861"  # pylint:disable=expression-not-assigned
    with pytest.raises(TypeError, match=r".*unsupported.*float"):
        86.1 * Shape(4, 2)  # pylint:disable=expression-not-assigned
    with pytest.raises(ValueError, match=r".*negative.*"):
        Shape(4, 2) * -2  # pylint:disable=expression-not-assigned


def test_shape_div():
    assert Shape() / 2 == Shape()
    with pytest.raises(ValueError, match=r".*negative.*"):
        Shape(1) / 2  # pylint:disable=expression-not-assigned
    assert Shape(0, is_null=True) / 10 == (0,)
    assert Shape(0, 0, 0, is_null=True) / 10 == (0, 0, 0)
    assert Shape(1, is_null=True) / 2 == Shape(0, is_null=True)
    assert Shape(2, 4) / 2 == Shape(1, 2)
    assert Shape(5, 10, 15) / 5 == Shape(1, 2, 3)
    with pytest.raises(TypeError, match=r".*unsupported.*str"):
        Shape(10, 8) / "861"  # pylint:disable=expression-not-assigned
    with pytest.raises(TypeError, match=r".*unsupported.*float"):
        Shape(10, 8) / 86.1  # pylint:disable=expression-not-assigned


def test_shape_prod():
    assert Shape().prod == 0
    assert Shape(1, 2, 3).prod == 6
    assert Shape(1).prod == 1
    shape = Shape(5, 6)
    assert shape.prod == 30
    assert shape.prod == 30


def test_shape_iter():
    assert list(Shape()) == []
    assert list(Shape(1, 2, 3)) == [1, 2, 3]


def test_shape_index():
    with pytest.raises(IndexError):
        Shape()[0]  # pylint:disable=expression-not-assigned
    with pytest.raises(IndexError):
        Shape(1, 2, 3)[10]  # pylint:disable=expression-not-assigned
    assert Shape(1, 2, 3)[1] == 2
    assert Shape(1, 2, 3)[:-1] == (1, 2)
    with pytest.raises(TypeError, match=r"must be.*not str"):
        Shape(1, 2, 3)["a"]  # pylint:disable=expression-not-assigned


def test_shape_equal():
    assert Shape() == Shape()
    assert Shape(1, 2, 3) == Shape(1, 2, 3)
    assert Shape((1, 2, 3)) == (1, 2, 3)
    assert Shape([1, 2, 3]) != [2, 3]
    assert Shape(1, 2, 3) != Shape(10)
    assert Shape(1) != 1
    assert Shape() != "test"


def test_shape_str():
    assert str(Shape()) == "Shape: "
    assert str(Shape(1)) == "Shape: 1"
    assert str(Shape(1, 2, 3)) == "Shape: 1x2x3"


def test_shape_repr():
    assert repr(Shape()) == "Shape(is_null=False)"
    assert repr(Shape(1)) == "Shape(1, is_null=False)"
    assert repr(Shape(1, 2, 3, is_null=True)) == "Shape(1, 2, 3, is_null=True)"
