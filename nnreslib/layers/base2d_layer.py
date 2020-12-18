# TODO: rename module name
import math
from typing import Any

from .base_layer import Layer
from ..utils.types import Shape


class Base2DLayer(Layer):
    """
    Basic class for 2D layers like convolution, maxpooling etc.
    """

    # pylint:disable=unused-argument
    def __init__(self, name: str, kernel: Shape, stride: Shape, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        if len(kernel) < 1:
            raise ValueError("'kernel' must be greater than 0")
        self.kernel = kernel
        if len(stride) < 1:
            raise ValueError("'stride' must be greater than 0")
        self.stride = stride

    def _set_output_shape(self, pad: Shape) -> Shape:
        return Shape(
            *(
                math.ceil((input_dim - kernel + 2 * pad) / stride + 1)
                for input_dim, kernel, pad, stride in zip(self.input_shape, self.kernel, pad, self.stride)
            )
        )
