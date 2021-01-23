from enum import Enum, auto, unique


@unique
class DTypes(Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"


@unique
class Backends(Enum):
    TF = auto()


DTYPE: DTypes = DTypes.FLOAT64
BACKEND: Backends = Backends.TF
