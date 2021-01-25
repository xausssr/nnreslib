from typing_extensions import TypedDict

SerializedMergeFunctionsType = str


class SerializedMergeInputsType(TypedDict):
    main_input: str
    merge_func: SerializedMergeFunctionsType
