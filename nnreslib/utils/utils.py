import importlib
import os


def load_all_modules_from_package(file_path: str, package: str) -> None:
    directory_name = os.path.dirname(file_path)
    for name in os.listdir(directory_name):
        if name == "__init__.py" or not name.endswith(".py"):
            continue
        importlib.import_module("." + name[:-3], package)
