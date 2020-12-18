from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model
    from .utils.types import Shape


class Settings:
    __slots__ = ("inputs", "outputs", "batch_size", "model")

    def __init__(self, inputs: Shape, outputs: Shape, batch_size: int, model: Model):
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.model = model
