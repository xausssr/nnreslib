from nnreslib.func import my_func


def test_my_func():
    input_value = 10
    result = my_func(input_value)

    assert result == input_value ** 2


def test_my_func2():
    input_value = 500
    result = my_func(input_value)

    assert result == input_value // 2
