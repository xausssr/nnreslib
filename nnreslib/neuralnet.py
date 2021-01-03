from .graph import ForwardGraph
from .model import Model
from .utils.metrics import Metrics


class NeuralNet:
    __slots__ = ("batch_size", "model", "forward_graph", "verbose", "metrics")

    def __init__(self, batch_size: int, model: Model, verbose: bool = False) -> None:
        """
        Build ANN
        settings - describe ANN architect
        verbose - print ANN info to console
        """
        self.batch_size = batch_size
        self.model = model
        self.forward_graph = self.build_graph()
        self.verbose = verbose
        self.metrics = Metrics()
        # self.weights: List[int] = []  # numpy.array

    def build_graph(self) -> ForwardGraph:
        return ForwardGraph(self.batch_size, self.model)

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
