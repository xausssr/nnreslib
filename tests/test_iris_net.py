import numpy as np

from nnreslib.architecture import ArchitectureType
from nnreslib.layers import FullyConnectedLayer, InputLayer
from nnreslib.model import Model
from nnreslib.utils.initialization import Initialization, StandartInitializer
from nnreslib.utils.types import Shape


def test_iris_net():
    data = np.load("./data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5, initializer=Initialization(StandartInitializer.ZEROS.value)),
        FullyConnectedLayer("fc_2", neurons=6, initializer=Initialization(StandartInitializer.ZEROS.value)),
        FullyConnectedLayer(
            "fc_3", neurons=3, initializer=Initialization(StandartInitializer.ZEROS.value), is_out=True
        ),
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "LevenbergMarquardt",
        x_train,
        y_train,
        x_validation,
        y_validation,
        100,
        0.061,
        step_into_epoch=10,
        regularisation_factor_init=3.0,
        regularisation_factor_decay=10.0,
        regularisation_factor_increase=10.0,
    )

    assert epoch == 94
    assert loss < 0.061
