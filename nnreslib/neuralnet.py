# from typing import List

from .settings import Settings
from .utils.metrics import Metrics


class NeuralNet:
    __slots__ = ("verbose", "settings", "metrics")

    def __init__(self, settings: Settings, verbose: bool = False) -> None:
        """
        Build ANN
        settings - describe ANN architect
        verbose - print ANN info to console
        """
        self.verbose = verbose
        self.settings = settings
        self.metrics = Metrics()
        # self.weights: List[int] = []  # numpy.array

    def build_graph(self) -> None:
        # call check graph validation
        print(self.verbose)

    def tune_graph(self) -> None:
        print(self.verbose)

    def init_graph(self) -> None:
        print(self.verbose)

    def train(self) -> None:
        # add part to computation graph for training
        # if visualisation - get callbacks or plot results with matplotlib
        # async
        print(self.verbose)

    def predict(self) -> None:
        # async
        print(self.verbose)

    def get_parametrs(self) -> None:
        # return numpy tensors with weights
        print(self.verbose)

    def save(self) -> None:
        # save graph with neural network to file
        print(self.verbose)

    def load(self) -> None:
        # load from file
        print(self.verbose)
