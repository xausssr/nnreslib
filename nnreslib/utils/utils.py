import importlib
import json
import os
from typing import Any, Dict

from jsonschema import ValidationError, validate


def load_all_modules_from_package(file_path: str, package: str) -> None:
    directory_name = os.path.dirname(file_path)
    for name in os.listdir(directory_name):
        if name == "__init__.py" or not name.endswith(".py"):
            continue
        importlib.import_module("." + name[:-3], package)


def validate_json(data: Dict[str, Any], schema_path: str) -> None:
    with open(schema_path, encoding="utf-8") as input_fd:
        schema = json.load(input_fd)

    validate(data, schema)


__all__ = ["load_all_modules_from_package", "validate_json", "ValidationError"]
