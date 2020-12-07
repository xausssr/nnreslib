from typing import List

import attr

from .utils.layers import Layer
from .utils.types import Shape


@attr.s(auto_attribs=True)
class Settings:
    outs: int
    batch_size: int
    inputs: Shape
    architecture: List[Layer]
