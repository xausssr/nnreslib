import numpy as np

from nnreslib.architecture import ArchitectureType
from nnreslib.layers import FullyConnectedLayer, InputLayer
from nnreslib.model import Model
from nnreslib.utils.initialization import Initialization
from nnreslib.utils.types import ActivationFunctions, Shape

seed = np.random.RandomState(42)


def he_normal(
    input_shape: Shape, output_shape: Shape, parameter_shape: Shape, data_mean: float = 0.0, data_std: float = 0.0
) -> np.ndarray:
    return seed.uniform(size=parameter_shape.dimension)


def test_iris_net():
    data = np.load("./data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5, initializer=Initialization(he_normal)),
        FullyConnectedLayer("fc_2", neurons=6, initializer=Initialization(he_normal)),
        FullyConnectedLayer(
            "fc_3",
            neurons=3,
            initializer=Initialization(he_normal),
            activation=ActivationFunctions.SOFT_MAX,
            is_out=True,
        ),
    ]

    model = Model(150, architecture)
    print(model.predict(x_train)[0][0])
    print(model.predict_proba(x_train)[0][0])
    epoch, loss = model.train(
        "LevenbergMarquardt",
        x_train,
        y_train,
        x_validation,
        y_validation,
        200,
        0.034,
        step_into_epoch=10,
        regularisation_factor_init=3.0,
        regularisation_factor_decay=10.0,
        regularisation_factor_increase=10.0,
    )

    assert epoch == 20
    assert loss < 0.02914
    assert np.array_equal(model.predict(x_train)[0][0], np.array([0, 0, 1]))
