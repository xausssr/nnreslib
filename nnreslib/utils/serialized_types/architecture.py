from typing import List, Tuple, Union

from typing_extensions import TypedDict

from .layer import SerializedLayerType

SerializedLayersListType = List[SerializedLayerType]


class SerializedLayersWithCustomInputsDefinition(TypedDict):
    inputs: Union[str, Tuple[str, ...]]
    layers: Union[SerializedLayerType, SerializedLayersListType]


SerializedLayersWithCustomInputs = List[SerializedLayersWithCustomInputsDefinition]
SerializedArchitectureLevelType = Union[SerializedLayerType, SerializedLayersListType, SerializedLayersWithCustomInputs]
SerializedArchitectureType = List[SerializedArchitectureLevelType]
