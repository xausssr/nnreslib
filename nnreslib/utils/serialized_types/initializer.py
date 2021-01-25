from typing import Union

from typing_extensions import TypedDict

SerializedStandartInitializerType = str


class SerializedCustomInitializerType(TypedDict):
    function: str
    # XXX: support serialize CustomInitializer


SerializedInitializeFunctionType = Union[
    SerializedStandartInitializerType,
    SerializedCustomInitializerType,
]


class SerializedInitializationType(TypedDict):
    weights_initializer: SerializedInitializeFunctionType
    biases_initializer: SerializedInitializeFunctionType
