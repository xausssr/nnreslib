from os import PathLike
from typing import Any, Dict, Union, overload

import numpy as np

from .architecture import Architecture, ArchitectureType
from .graph import ForwardGraph
from .graph.fit_graph.fit_graph import FitGraph  # FIXME: move FitGraph to above module
from .utils.metrics import Metrics


class Model:
    __slots__ = ("batch_size", "architecture", "forward_graph", "fit_graphs", "verbose", "metrics")

    @overload
    def __init__(
        self,
        batch_size: int,
        architecture: ArchitectureType,
        data_mean: float = 0.0,
        data_std: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """
        Load explicit defined architecture
        """

    @overload
    def __init__(
        self, batch_size: int, architecture: str, data_mean: float = 0.0, data_std: float = 0.0, verbose: bool = False
    ) -> None:
        """
        Load architecture from file
        """

    @overload
    def __init__(
        self,
        batch_size: int,
        architecture: PathLike,
        data_mean: float = 0.0,
        data_std: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """
        Load architecture from file
        """

    def __init__(
        self,
        batch_size: int,
        architecture: Union[ArchitectureType, str, PathLike],
        data_mean: float = 0.0,
        data_std: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """
        Build ANN
        verbose - print ANN info to console
        """
        self.batch_size = batch_size
        self.architecture = Architecture(architecture)
        self.architecture.initialize(data_mean, data_std)
        self.forward_graph = self.build_graph()
        self.fit_graphs: Dict[str, FitGraph] = {}
        self.verbose = verbose
        self.metrics = Metrics()
        # self.weights: List[int] = []  # numpy.array

    def build_graph(self) -> ForwardGraph:
        return ForwardGraph(self.batch_size, self.architecture)

    def tune_graph(self) -> None:
        ...

    def init_graph(self) -> None:
        ...

    def train(
        self,
        method: str,
        train_x_data: np.ndarray,
        train_y_data: np.ndarray,
        valid_x_data: np.ndarray,
        valid_y_data: np.ndarray,
        batch_size: int,
        max_epoch: int = 100,
        min_error: float = 1e-10,
        shuffle: bool = True,
        logging_step: int = 10,
        **kwargs: Any
    ) -> None:
        fit_graph = FitGraph.get_fitter(method)(
            self.batch_size, self.architecture, self.forward_graph
        )  # FIXME: create fitter in get_fitter
        self.fit_graphs[method] = fit_graph
        fit_graph.fit(
            train_x_data,
            train_y_data,
            valid_x_data,
            valid_y_data,
            batch_size,
            max_epoch,
            min_error,
            shuffle,
            logging_step,
            **kwargs
        )

    def predict(self) -> None:
        # async
        ...

    def get_parametrs(self) -> None:
        # return numpy tensors with weights
        ...

    def save(self) -> None:
        # save graph with neural network to file
        ...

    def load(self) -> None:
        # load from file
        ...
