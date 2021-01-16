from nnreslib.utils.merge import MergeInputs


def test_serialize():
    assert MergeInputs().serialize() == dict(main_input="", merge_func="RESHAPE_TO_MAIN")
    assert MergeInputs("l1").serialize() == dict(main_input="l1", merge_func="RESHAPE_TO_MAIN")
