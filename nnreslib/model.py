from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Tuple, Union, overload

import numpy as np

from .architecture import Architecture, ArchitectureType
from .graph import ForwardGraph
from .graph.fit_graph.fit_graph import FitGraph  # FIXME: move FitGraph to above module
from .utils.metrics import Metrics
from .utils.serialized_types import SerializedModelType
from .utils.utils import validate_json


class Model:
    __slots__ = ("batch_size", "architecture", "forward_graph", "fit_graphs", "verbose", "metrics")

    def __init__(
        self,
        batch_size: int,
        architecture: ArchitectureType,
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

    @overload
    @classmethod
    def from_json(cls, file: str) -> Model:
        ...

    @overload
    @classmethod
    def from_json(cls, file: PathLike) -> Model:
        ...

    @classmethod
    def from_json(cls, file: Union[str, PathLike]) -> Model:
        model_path = Path(file)
        with open(model_path, encoding="utf-8") as input_fd:
            model_def: SerializedModelType = json.load(input_fd)
        validate_json(model_def, "../doc/model.schema.json")  # TODO: get it from package
        return cls(
            model_def["batch_size"],
            Architecture.from_json(model_def["architecture"]["architecture"], model_def["architecture"]["is_built"]),
        )

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
        max_epoch: int = 100,
        min_error: float = 1e-10,
        shuffle: bool = True,
        logging_step: int = 10,
        **kwargs: Any
    ) -> Tuple[int, float]:
        fit_graph = FitGraph.get_fitter(method)(
            self.batch_size, self.architecture, self.forward_graph
        )  # FIXME: create fitter in get_fitter
        self.fit_graphs[method] = fit_graph
        return fit_graph.fit(
            train_x_data,
            train_y_data,
            valid_x_data,
            valid_y_data,
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

    def serialize(self, built: bool = False) -> SerializedModelType:
        serialized_data: SerializedModelType = dict(
            batch_size=self.batch_size,
            architecture=dict(architecture=self.architecture.serialize(built), is_built=built),
        )
        return serialized_data
