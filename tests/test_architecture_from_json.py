from json import JSONDecodeError

import pytest
from jsonschema.exceptions import ValidationError

from nnreslib.architecture import Architecture


def test_architecture_not_json(tmp_path):
    arch_file = tmp_path / "test_arch.yaml"
    arch_file.write_text("")
    with pytest.raises(ValueError, match="supports only json"):
        Architecture(arch_file)


def test_architecture_json_validation_empty_file(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text("")
    with pytest.raises(JSONDecodeError, match=r"Expecting value: line 1 column 1 \(char 0\)"):
        Architecture(arch_file)


def test_architecture_json_validation_empty_root(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text("{}")
    with pytest.raises(ValidationError, match="'architecture' is a required property"):
        Architecture(arch_file)


def test_architecture_json_validation_empty_architecture(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text("""{"architecture":{}}""")
    with pytest.raises(ValidationError, match="does not have enough properties"):
        Architecture(arch_file)


def test_architecture_json_validation_low_layers(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "q": {
            "type": "Flatten"
        }
    }
}"""
    )
    with pytest.raises(ValidationError, match="does not have enough properties"):
        Architecture(arch_file)


def test_architecture_json_validation_empty_layer_name(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "": {
            "type": "Flatten"
        },
        "flat": {
            "type": "Flatten"
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="does not match any of the regexes"):
        Architecture(arch_file)


# cannot verify unique layer names with schema validation


def test_architecture_json_validation_layer_without_type(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "input": {
            "input_shape": [
                15,
                10,
                1
            ]
        },
        "flat": {
            "type": "Flatten"
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'type' is a required property"):
        Architecture(arch_file)

    arch_file.write_text(
        """
{
    "architecture": {
        "input": {
            "type": "Input",
            "input_shape": [
                15,
                10,
                1
            ]
        },
        "flat": {
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'type' is a required property"):
        Architecture(arch_file)


# FIXME: need fix jsonschema.validate in architecture.py
def skip_test_architecture_json_validation_input_layer_without_input_shape(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "input": {
            "type": "Input"
        },
        "flat": {
            "type": "Flatten"
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'input_shape' is a required property"):
        Architecture(arch_file)


# FIXME: need fix jsonschema.validate in architecture.py
def skip_test_architecture_json_validation_conv_layers_without_kernel(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten"
        },
        "conv_1": {
            "type": "Convolution",
            "stride": [
                5,
                5
            ],
            "filters": 3
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'kernel' is a required property"):
        Architecture(arch_file)

    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten"
        },
        "pool_1": {
            "type": "MaxPool",
            "stride": [
                5,
                5
            ]
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'kernel' is a required property"):
        Architecture(arch_file)


# FIXME: need fix jsonschema.validate in architecture.py
def skip_test_architecture_json_validation_conv_layers_without_stride(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten"
        },
        "conv_1": {
            "type": "Convolution",
            "kernel": [
                5,
                5
            ],
            "filters": 3
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'stride' is a required property"):
        Architecture(arch_file)

    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten"
        },
        "pool_1": {
            "type": "MaxPool",
            "kernel": [
                5,
                5
            ]
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'stride' is a required property"):
        Architecture(arch_file)


# FIXME: need fix jsonschema.validate in architecture.py
def skip_test_architecture_json_validation_convolution_layer_without_filters(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten"
        },
        "conv_1": {
            "type": "Convolution",
            "kernel": [
                5,
                5
            ],
            "stride": [
                5,
                5
            ]
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'filters' is a required property"):
        Architecture(arch_file)


# FIXME: need fix jsonschema.validate in architecture.py
def skip_test_architecture_json_validation_fullyconnected_layer_without_neurons(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten"
        },
        "fc_1": {
            "type": "FullyConnected"
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="'neurons' is a required property"):
        Architecture(arch_file)


def test_architecture_json_validation_zero_filters(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten"
        },
        "conv_1": {
            "type": "Convolution",
            "kernel": [
                5,
                5
            ],
            "stride": [
                5,
                5
            ],
            "filters": 0
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="is less than the minimum of 1"):
        Architecture(arch_file)


def test_architecture_json_validation_zero_neurons(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten"
        },
        "fc_1": {
            "type": "FullyConnected",
            "neurons": 0
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="is less than the minimum of 1"):
        Architecture(arch_file)


def test_architecture_json_validation_empty_shape(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "input": {
            "type": "Input",
            "input_shape": []
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="is too short"):
        Architecture(arch_file)

    arch_file.write_text(
        """
{
    "architecture": {
        "conv_1": {
            "type": "Convolution",
            "kernel": [
            ],
            "stride": [
                5,
                5
            ],
            "filters": 1
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="is too short"):
        Architecture(arch_file)


def test_architecture_json_validation_incorrect_type(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "XXX"
        }
    }
}
"""
    )
    with pytest.raises(
        ValidationError, match=r"is not one of \['Input', 'Convolution', 'MaxPool', 'Flatten', 'FullyConnected'\]"
    ):
        Architecture(arch_file)


def test_architecture_json_validation_additional_layers_parameters(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "input": {
            "type": "Input",
            "input_shape": [
                15,
                10,
                1
            ],
            "some_param": 1
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        Architecture(arch_file)

    arch_file.write_text(
        """
{
    "architecture": {
        "conv_1": {
            "type": "Convolution",
            "kernel": [
                5,
                5
            ],
            "stride": [
                5,
                5
            ],
            "filters": 3,
            "some_param": 1
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        Architecture(arch_file)

    arch_file.write_text(
        """
{
    "architecture": {
        "pool_1": {
            "type": "MaxPool",
            "kernel": [
                5,
                5
            ],
            "stride": [
                5,
                5
            ],
            "some_param": 1
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        Architecture(arch_file)

    arch_file.write_text(
        """
{
    "architecture": {
        "flat": {
            "type": "Flatten",
            "some_param": 1
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        Architecture(arch_file)

    arch_file.write_text(
        """
{
    "architecture": {
        "fc": {
            "type": "FullyConnected",
            "neurons": 2,
            "some_param": 1
        }
    }
}
"""
    )
    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        Architecture(arch_file)


def test_architecture_json_flat_valid(tmp_path):
    arch_file = tmp_path / "test_arch.json"
    arch_file.write_text(
        """
{
    "architecture": {
        "input": {
            "type": "Input",
            "input_shape": [
                15,
                10,
                1
            ]
        },
        "conv_1": {
            "type": "Convolution",
            "kernel": [
                5,
                5
            ],
            "stride": [
                5,
                5
            ],
            "filters": 3
        },
        "pool_1": {
            "type": "MaxPool",
            "kernel": [
                3,
                2
            ],
            "stride": [
                1,
                1
            ]
        },
        "flat_1": {
            "type": "Flatten"
        },
        "fc_1": {
            "type": "FullyConnected",
            "neurons": 3
        },
        "fc_2": {
            "type": "FullyConnected",
            "neurons": 2
        }
    }
}
"""
    )
    simple_architecture = Architecture(arch_file)
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
    assert simple_architecture._architecture == {  # pylint:disable=protected-access
        "conv_1": ("input",),
        "pool_1": ["conv_1"],
        "flat_1": ["pool_1"],
        "fc_1": ["flat_1"],
        "fc_2": ["fc_1"],
    }
