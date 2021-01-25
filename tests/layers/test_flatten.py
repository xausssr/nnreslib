from nnreslib.layers import FlattenLayer


def test_serialize():
    assert FlattenLayer("f_1").serialize() == dict(name="f_1", type="FlattenLayer", merge=None, is_out=False)
    assert FlattenLayer("f_2").serialize() == dict(name="f_2", type="FlattenLayer", merge=None, is_out=False)
