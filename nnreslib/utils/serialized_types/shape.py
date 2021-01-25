from typing import List, Union

from typing_extensions import TypedDict

SerializedSimpleShapeType = List[int]


class SerializedFullShapeType(TypedDict):
    shape: SerializedSimpleShapeType
    is_null: bool


SerializedShapeType = Union[
    SerializedSimpleShapeType,
    SerializedFullShapeType,
]
