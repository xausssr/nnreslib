from abc import ABC
from typing import Optional, Tuple

import numpy as np

from ..backend import graph as G


class Graph(ABC):

    __slots__ = ("batch_size", "session", "__weakref__")

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.session = G.Session()

    # FIXME input data is List[np.ndarray] or np.ndarray
    def _prepare_batch(
        self, x_data: np.ndarray, y_data: Optional[np.ndarray] = None, shuffle: bool = True
    ) -> G.Dataset:
        dataset = G.Dataset.from_tensor_slices(x_data)
        if y_data is not None:
            y_dataset = G.Dataset.from_tensor_slices(y_data)
            dataset = G.Dataset.zip((dataset, y_dataset))
        if shuffle:
            dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)
        return dataset.batch(self.batch_size, drop_remainder=True)

    @G.graph_function  # type: ignore
    def _get_batches(self, data: G.Dataset, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        tensor_array_x = G.TensorArray(size=0, dynamic_size=True)
        tensor_array_y = G.TensorArray(size=0, dynamic_size=True)
        i = 0
        for batch in data:
            if train:
                batch_x, batch_y = batch
                tensor_array_x = tensor_array_x.write(i, batch_x)
                tensor_array_y = tensor_array_y.write(i, batch_y)
            else:
                tensor_array_x = tensor_array_x.write(i, batch)
            i += 1
        if train:
            return tensor_array_x.stack(), tensor_array_y.stack()
        batches = tensor_array_x.stack()
        return batches, batches
