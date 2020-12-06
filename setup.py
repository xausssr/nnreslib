import re
from typing import List

from setuptools import find_packages, setup


def get_version() -> str:
    file = "nnreslib/__init__.py"
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'^__version__\s*\=\s*[\'"]([^\'""]+)[\'"]', content)
    return match.group(1)


def get_requirements() -> List[str]:
    ret: List[str] = []
    with open("requirements/main.txt", encoding="utf-8") as input_fd:
        for line in input_fd:
            ret.append(line.rstrip())
    return ret


setup(
    version=get_version(),
    packages=find_packages(include=["nnreslib", "nnreslib.*"]),
    package_data={"nnreslib": ["py.typed"]},
    install_requires=get_requirements(),
)
