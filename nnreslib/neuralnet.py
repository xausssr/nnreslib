from typing import Dict, List, Type

from .settings import Settings
from .utils import log
from .utils.layers import ConvolutionLayer, FlattenLayer, Layer, MaxPoolLayer
from .utils.types import Shape

logger = log.get(__name__)


class NeuralNet:
    def __init__(self, settings: Settings, verbose: bool = False) -> None:
        """
        Build ANN
        settings - describe ANN architect
        verbose - print ANN info to console
        """
        self.verbose = verbose
        self.settings = settings
        self.metrics: Dict[str, int] = {}  # history of train: dict
        self.weights: List[int] = []  # numpy.array

    def check_arch(self) -> bool:
        # Check correct of graph (for example: convolution out shape not negative)
        layers = self.settings.architecture
        last_filters_count = self.settings.inputs[-1]
        pre_layer_type: Type[Layer] = ConvolutionLayer
        pre_layer_inputs = self.settings.inputs
        for layer in layers:
            if not (
                isinstance(layer, (ConvolutionLayer, MaxPoolLayer, FlattenLayer))
                and pre_layer_type in (ConvolutionLayer, MaxPoolLayer)
            ):
                return False
            pad = Shape()
            if isinstance(layer, ConvolutionLayer):
                pad = 2 * layer.pad
            layer.out_shape = layer.make_out_shape(pre_layer_inputs, pad)
            if not layer.check_output_shape():
                logger.warning("You have zero-shape layer, check you model around %s", layer.name)
                return False
            if isinstance(layer, ConvolutionLayer):
                last_filters_count = layer.filters
            if isinstance(layer, FlattenLayer):
                layer.out_shape = Shape(pre_layer_inputs.prod() * last_filters_count)

            pre_layer_type = type(layer)
            pre_layer_inputs = layer.out_shape

        return True

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
