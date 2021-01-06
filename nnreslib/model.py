from os import PathLike
from typing import Union, overload

from .architecture import Architecture, ArchitectureType
from .graph import ForwardGraph
from .utils.metrics import Metrics


class Model:
    __slots__ = ("batch_size", "architecture", "forward_graph", "verbose", "metrics")

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
        self.verbose = verbose
        self.metrics = Metrics()
        # self.weights: List[int] = []  # numpy.array

    def build_graph(self) -> ForwardGraph:
        return ForwardGraph(self.batch_size, self.architecture)

    def tune_graph(self) -> None:
        ...

    def init_graph(self) -> None:
        ...

    def train(self) -> None:
        # add part to computation graph for training
        # if visualisation - get callbacks or plot results with matplotlib
        # async
        ...

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
