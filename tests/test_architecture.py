import pytest

from nnreslib.architecture import Architecture, ArchitectureType
from nnreslib.layers import ConvolutionLayer, FlattenLayer, FullyConnectedLayer, InputLayer, MaxPoolLayer
from nnreslib.utils.merge import MergeInputs
from nnreslib.utils.types import Shape


def test_flat_valid():
    simple_architecture_def: ArchitectureType = [
        InputLayer("input", Shape(15, 10, 1)),
        ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 3),
        MaxPoolLayer("pool_1", Shape(3, 2), Shape(1, 1)),
        FlattenLayer("flat_1"),
        FullyConnectedLayer("fc_1", 3),
        FullyConnectedLayer("fc_2", 2),
    ]
    simple_architecture = Architecture(simple_architecture_def)
    # pylint:disable=protected-access
    real_shapes = [(x.layer.input_shape, x.layer.output_shape) for x in simple_architecture._layers.values()]
    for real_shape, true_shape in zip(
        real_shapes,
        [
            ((15, 10, 1), (15, 10, 1)),
            ((15, 10, 1), (3, 2, 3)),
            ((3, 2, 3), (1, 1, 3)),
            ((1, 1, 3), (3,)),
            ((3,), (3,)),
            ((3,), (2,)),
        ],
    ):
        assert real_shape[0] == true_shape[0] and real_shape[1] == true_shape[1]
    assert simple_architecture._architecture == [  # pylint:disable=protected-access
        ("conv_1", ("input",)),
        ("pool_1", ("conv_1",)),
        ("flat_1", ("pool_1",)),
        ("fc_1", ("flat_1",)),
        ("fc_2", ("fc_1",)),
    ]


def test_inception_valid():
    simple_architecture_def: ArchitectureType = [
        InputLayer("input", Shape(15, 10, 1)),
        (
            ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 5),
            ConvolutionLayer("conv_2", Shape(1, 1), Shape(1, 1), 5),
        ),
        {
            ("conv_1", "conv_2"): MaxPoolLayer("pool_1", Shape(3, 2), Shape(2, 2), merge=MergeInputs()),
        },
        {
            "pool_1": FlattenLayer("flat_1"),
        },
        FullyConnectedLayer("fc_1", 3),
        FullyConnectedLayer("fc_2", 2),
    ]
    simple_architecture = Architecture(simple_architecture_def)
    # pylint:disable=protected-access
    real_shapes = [(x.layer.input_shape, x.layer.output_shape) for x in simple_architecture._layers.values()]
    for real_shape, true_shape in zip(
        real_shapes,
        [
            # input_shape  output_shape
            ((15, 10, 1), (15, 10, 1)),  # input
            ((15, 10, 1), (3, 2, 5)),  # conv_1
            ((15, 10, 1), (15, 10, 5)),  # conv_2
            ((3, 2, 10), (1, 1, 10)),  # pool_1
            ((1, 1, 10), (10,)),  # flat_1
            ((10,), (3,)),  # fc_1
            ((3,), (2,)),  # fc_2
        ],
    ):
        assert real_shape[0] == true_shape[0] and real_shape[1] == true_shape[1]
    assert simple_architecture._architecture == [  # pylint:disable=protected-access
        ("conv_1", ("input",)),
        ("conv_2", ("input",)),
        ("pool_1", ("conv_1", "conv_2")),
        ("flat_1", ("pool_1",)),
        ("fc_1", ("flat_1",)),
        ("fc_2", ("fc_1",)),
    ]


# TODO: add test for valid arch with padding
# TODO: add test for network with input layer in the middle of the definition
# BUG: there may be an error here ^^^^^


def test_bad_arch_definition():
    # first not dict
    bad_arch_1 = [{"test_1": InputLayer("input", Shape(250, 250, 3))}]
    with pytest.raises(ValueError, match="not Mapping"):
        Architecture(bad_arch_1)

    # fist must be InputLayer
    bad_arch_2 = [ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 5)]
    with pytest.raises(ValueError, match="InputLayer"):
        Architecture(bad_arch_2)

    # unique names
    bad_arch_3 = [InputLayer("t1", Shape(250, 250, 3)), ConvolutionLayer("t1", Shape(5, 5), Shape(5, 5), 5)]
    with pytest.raises(ValueError, match="unique: t1"):
        Architecture(bad_arch_3)

    # bad sequence
    bad_arch_4 = [
        InputLayer("input", Shape(250, 250, 3)),
        ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 5),
        FullyConnectedLayer("fc_1", 3),
    ]
    with pytest.raises(ValueError, match=r"Unsupported.*conv_1 -> fc_1"):
        Architecture(bad_arch_4)
    bad_arch_5 = [
        InputLayer("input", Shape(250)),
        FullyConnectedLayer("fc_1", 3),
        ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 5),
    ]
    with pytest.raises(ValueError, match=r"Unsupported.*fc_1 -> conv_1"):
        Architecture(bad_arch_5)

    # all inputs is feedback
    bad_arch_6 = [
        InputLayer("input", Shape(250, 250, 3)),
        {"conv_2": ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 5)},
        ConvolutionLayer("conv_2", Shape(5, 5), Shape(5, 5), 5),
    ]
    with pytest.raises(ValueError, match=r"feedback.*conv_1"):
        Architecture(bad_arch_6)

    # merge for non trivial input
    bad_arch_7 = [
        InputLayer("input", Shape(250, 250, 3)),
        (
            ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 5),
            ConvolutionLayer("conv_2", Shape(1, 1), Shape(1, 1), 5),
        ),
        MaxPoolLayer("pool_1", Shape(3, 2), Shape(2, 2)),
    ]
    with pytest.raises(ValueError, match=r"multiple.*merge.*pool_1"):
        Architecture(bad_arch_7)

    # merge with non-depend main input
    bad_arch_8 = [
        InputLayer("input", Shape(15, 10, 1)),
        (
            ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 5),
            ConvolutionLayer("conv_2", Shape(1, 1), Shape(1, 1), 5),
        ),
        {
            ("conv_1", "conv_2"): MaxPoolLayer("pool_1", Shape(3, 2), Shape(2, 2), merge=MergeInputs("input")),
        },
    ]
    with pytest.raises(ValueError, match=r"Layer 'pool_1'.*depend.*input"):
        Architecture(bad_arch_8)


def test_serialize():
    flat_architecture_def: ArchitectureType = [
        InputLayer("input", Shape(15, 10, 1)),
        ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 3),
        MaxPoolLayer("pool_1", Shape(3, 2), Shape(1, 1)),
        FlattenLayer("flat_1"),
        FullyConnectedLayer("fc_1", 3),
        FullyConnectedLayer("fc_2", 2),
    ]
    flat_architecture = Architecture(flat_architecture_def)
    assert flat_architecture.serialize() == [
        {"name": "input", "type": "InputLayer", "input_shape": [15, 10, 1]},
        {
            "name": "conv_1",
            "type": "ConvolutionLayer",
            "merge": None,
            "is_out": False,
            "activation": "RELU",
            "initializer": {"weights_initializer": "HE_NORMAL", "biases_initializer": "ZEROS"},
            "kernel": [5, 5],
            "stride": [5, 5],
            "filters": 3,
            "pad": {"shape": [0, 0], "is_null": True},
        },
        {"name": "pool_1", "type": "MaxPoolLayer", "merge": None, "is_out": False, "kernel": [3, 2], "stride": [1, 1]},
        {"name": "flat_1", "type": "FlattenLayer", "merge": None, "is_out": False},
        {
            "name": "fc_1",
            "type": "FullyConnectedLayer",
            "merge": None,
            "is_out": False,
            "activation": "SIGMOID",
            "initializer": {"weights_initializer": "HE_NORMAL", "biases_initializer": "ZEROS"},
            "neurons": 3,
        },
        {
            "name": "fc_2",
            "type": "FullyConnectedLayer",
            "merge": None,
            "is_out": False,
            "activation": "SIGMOID",
            "initializer": {"weights_initializer": "HE_NORMAL", "biases_initializer": "ZEROS"},
            "neurons": 2,
        },
    ]

    inception_architecture_def: ArchitectureType = [
        InputLayer("input", Shape(15, 10, 1)),
        (
            ConvolutionLayer("conv_1", Shape(5, 5), Shape(5, 5), 5),
            ConvolutionLayer("conv_2", Shape(1, 1), Shape(1, 1), 5),
        ),
        {
            ("conv_1", "conv_2"): MaxPoolLayer("pool_1", Shape(3, 2), Shape(2, 2), merge=MergeInputs()),
        },
        {
            "pool_1": FlattenLayer("flat_1"),
        },
        FullyConnectedLayer("fc_1", 3),
        FullyConnectedLayer("fc_2", 2),
    ]
    inception_architecture = Architecture(inception_architecture_def)
    assert inception_architecture.serialize() == [
        {"name": "input", "type": "InputLayer", "input_shape": [15, 10, 1]},
        [
            {
                "name": "conv_1",
                "type": "ConvolutionLayer",
                "merge": None,
                "is_out": False,
                "activation": "RELU",
                "initializer": {"weights_initializer": "HE_NORMAL", "biases_initializer": "ZEROS"},
                "kernel": [5, 5],
                "stride": [5, 5],
                "filters": 5,
                "pad": {"shape": [0, 0], "is_null": True},
            },
            {
                "name": "conv_2",
                "type": "ConvolutionLayer",
                "merge": None,
                "is_out": False,
                "activation": "RELU",
                "initializer": {"weights_initializer": "HE_NORMAL", "biases_initializer": "ZEROS"},
                "kernel": [1, 1],
                "stride": [1, 1],
                "filters": 5,
                "pad": {"shape": [0, 0], "is_null": True},
            },
        ],
        {
            "name": "pool_1",
            "type": "MaxPoolLayer",
            "merge": {"main_input": "conv_1", "merge_func": "RESHAPE_TO_MAIN"},
            "is_out": False,
            "kernel": [3, 2],
            "stride": [2, 2],
        },
        {"name": "flat_1", "type": "FlattenLayer", "merge": None, "is_out": False},
        {
            "name": "fc_1",
            "type": "FullyConnectedLayer",
            "merge": None,
            "is_out": False,
            "activation": "SIGMOID",
            "initializer": {"weights_initializer": "HE_NORMAL", "biases_initializer": "ZEROS"},
            "neurons": 3,
        },
        {
            "name": "fc_2",
            "type": "FullyConnectedLayer",
            "merge": None,
            "is_out": False,
            "activation": "SIGMOID",
            "initializer": {"weights_initializer": "HE_NORMAL", "biases_initializer": "ZEROS"},
            "neurons": 2,
        },
    ]
