from typing import Dict, List, Tuple, Union

from typing_extensions import TypedDict

from .layer import SerializedLayerType

SerializedLayersListType = List[SerializedLayerType]


class SerializedLayersWithCustomInputsDefinition(TypedDict):
    inputs: Union[str, Tuple[str, ...]]
    layers: Union[SerializedLayerType, SerializedLayersListType]


SerializedLayersWithCustomInputs = List[SerializedLayersWithCustomInputsDefinition]
SerializedArchitectureLevelType = Union[SerializedLayerType, SerializedLayersListType, SerializedLayersWithCustomInputs]
SerializedNotBuiltArchitectureType = List[SerializedArchitectureLevelType]


class SerializedLayerInfoType(TypedDict):
    layer_id: int
    layer: SerializedLayerType


class SerializedArchitectureInfoType(TypedDict):
    layer: str
    inputs: List[str]


class SerializedBuiltArchitectureType(TypedDict):
    neurons_count: int
    _layers: Dict[str, SerializedLayerInfoType]
    _input_layers: List[str]
    _output_layers: List[str]
    _trainable_layers: List[str]
    _architecture: List[SerializedArchitectureInfoType]


SerializedArchitectureType = Union[SerializedNotBuiltArchitectureType, SerializedBuiltArchitectureType]
