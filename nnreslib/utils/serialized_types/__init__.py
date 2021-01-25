from typing import Optional

from typing_extensions import TypedDict

from .initializer import SerializedInitializationType, SerializedInitializeFunctionType
from .merge import SerializedMergeInputsType
from .shape import SerializedShapeType


class SerializedLayerType(TypedDict, total=False):
    name: str
    type: str  # noqa
    merge: Optional[SerializedMergeInputsType]
    is_out: bool
    input_shape: SerializedShapeType
    kernel: SerializedShapeType
    stride: SerializedShapeType
    activation: str
    initializer: SerializedInitializationType
    filters: int
    pad: SerializedShapeType
    neurons: int


__all__ = [
    "SerializedInitializationType",
    "SerializedInitializeFunctionType",
    "SerializedLayerType",
    "SerializedMergeInputsType",
    "SerializedShapeType",
]
