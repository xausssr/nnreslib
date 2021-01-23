import math
from typing import Any, Optional

from .base_layer import Layer
from ..utils.merge import MergeInputs
from ..utils.types import Shape


class BaseConvLayer(Layer):
    """
    Basic class for 2D layers like convolution, maxpooling etc.
    """

    def __init__(
        self,
        name: str,
        kernel: Shape,
        stride: Shape,
        merge: Optional[MergeInputs] = None,
        is_out: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, merge=merge, is_out=is_out, **kwargs)
        if len(kernel) < 1:
            raise ValueError("'kernel' must be greater than 0")
        self.kernel = kernel
        if len(stride) < 1:
            raise ValueError("'stride' must be greater than 0")
        self.stride = stride

    def _calc_output_shape(self, pad: Optional[Shape] = None, last_dim: int = -1) -> Shape:
        if pad is None:
            pad = Shape((0,) * len(self.kernel), is_null=True)
        if last_dim == -1:
            last_dim = self.input_shape[-1]
        return Shape(
            *(
                math.ceil((input_dim - kernel + 2 * pad) / stride + 1)
                for input_dim, kernel, pad, stride in zip(self.input_shape[:-1], self.kernel, pad, self.stride)
            ),
            last_dim,
        )
