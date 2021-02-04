from abc import ABC
from typing import Generator, Iterable, Optional, Tuple, Union, overload

import numpy as np

from ..backend import graph as G


class Graph(ABC):

    __slots__ = ("batch_size", "session", "__weakref__")

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.session = G.Session()

    # FIXME input data is List[np.ndarray] or np.ndarray
    def _prepare_batch(
        self, x_data: np.ndarray, y_data: Optional[Union[np.ndarray, Iterable[np.ndarray]]] = None, shuffle: bool = True
    ) -> G.Dataset:
        dataset = G.Dataset.from_tensor_slices(x_data)
        if y_data is not None:
            y_dataset: Tuple[G.Dataset, ...]
            if isinstance(y_data, np.ndarray):
                y_dataset = (G.Dataset.from_tensor_slices(y_data),)
            else:
                y_dataset = tuple(G.Dataset.from_tensor_slices(y) for y in y_data)
            dataset = G.Dataset.zip((dataset, y_dataset))
        if shuffle:
            dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)
        return dataset.batch(self.batch_size, drop_remainder=True)

    @overload
    def _get_batches(self, data: G.Dataset) -> Generator[Tuple[np.ndarray], None, None]:
        ...

    @overload
    def _get_batches(
        self, data: G.Dataset, is_train: bool
    ) -> Generator[Tuple[np.ndarray, Tuple[np.ndarray, ...]], None, None]:
        ...

    def _get_batches(
        self, data: G.Dataset, is_train: bool = True  # pylint:disable=unused-argument
    ) -> Generator[Union[Tuple[np.ndarray], Tuple[np.ndarray, Tuple[np.ndarray, ...]]], None, None]:
        batch_iterator = data.make_one_shot_iterator().get_next()
        try:
            while True:
                yield self.session.run(batch_iterator)
        except G.OutOfRangeError:
            pass
