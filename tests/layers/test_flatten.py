from nnreslib.layers import FlattenLayer
from nnreslib.utils.merge import MergeInputs


def test_serialize():
    assert FlattenLayer("f_1").serialize() == dict(name="f_1", type="FlattenLayer", merge=None, is_out=False)
    assert FlattenLayer("f_1", is_out=True).serialize() == dict(
        name="f_1", type="FlattenLayer", merge=None, is_out=True
    )
    assert FlattenLayer("f_1", merge=MergeInputs()).serialize() == dict(
        name="f_1",
        type="FlattenLayer",
        merge=dict(main_input="", merge_func="RESHAPE_TO_MAIN"),
        is_out=False,
    )
    assert FlattenLayer("f_1", merge=MergeInputs(), is_out=True).serialize() == dict(
        name="f_1",
        type="FlattenLayer",
        merge=dict(main_input="", merge_func="RESHAPE_TO_MAIN"),
        is_out=True,
    )
