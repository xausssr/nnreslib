from typing import Union

from typing_extensions import TypedDict

from .architecture import SerializedBuiltArchitectureType, SerializedNotBuiltArchitectureType


class SerializedModelArchitectureType(TypedDict):
    architecture: Union[SerializedBuiltArchitectureType, SerializedNotBuiltArchitectureType]
    is_built: bool


class SerializedModelType(TypedDict):
    batch_size: int
    architecture: SerializedModelArchitectureType
