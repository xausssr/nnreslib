# import matplotlib.pyplot as plt
# from nnreslib.utils.metrics import OpMode

import numpy as np
import tensorflow as tf

from nnreslib.architecture import ArchitectureType
from nnreslib.layers import FullyConnectedLayer, InputLayer
from nnreslib.model import Model
from nnreslib.utils.types import ActivationFunctions, Shape


def test_iris_2_outs_net():
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)

    data = np.load("./tests/data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)
    y_train = [y_train[:, :-1], y_train[:, -1:]]

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)
    y_validation = [y_validation[:, :-1], y_validation[:, -1:]]

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        [FullyConnectedLayer("fc_2_1", neurons=6), FullyConnectedLayer("fc_2_2", neurons=3)],
        {
            "fc_2_1": FullyConnectedLayer(
                "fc_3_1",
                neurons=2,
                activation=ActivationFunctions.SOFTMAX,
                is_out=True,
            ),
            "fc_2_2": FullyConnectedLayer(
                "fc_3_2",
                neurons=1,
                is_out=True,
            ),
        },
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "LevenbergMarquardt",
        x_train,
        y_train,
        x_validation,
        y_validation,
        10,
        0.0005,
        step_into_epoch=10,
        regularisation_factor_init=3.0,
        regularisation_factor_decay=7.0,
        regularisation_factor_increase=7.0,
    )

    assert epoch == 10
    assert loss < 0.1
    # assert np.array_equal(model.predict(x_train)[0], np.array([1, 0, 0]))

    # Only for interactive testing
    # plt.plot(model.metrics.results[OpMode.TRAIN]["MSE"], label="Train MSE")
    # plt.plot(model.metrics.results[OpMode.TRAIN]["RMSE"], label="Train RMSE")
    # plt.plot(model.metrics.results[OpMode.TRAIN]["MAE"], label="Train MAE")
    # plt.plot(model.metrics.results[OpMode.TRAIN]["CCE"], label="Train CCE")
    # plt.plot(model.metrics.results[OpMode.VALID]["MSE"], label="Valid MSE")
    # plt.plot(model.metrics.results[OpMode.VALID]["RMSE"], label="Valid RMSE")
    # plt.plot(model.metrics.results[OpMode.VALID]["MAE"], label="Valid MAE")
    # plt.plot(model.metrics.results[OpMode.VALID]["CCE"], label="Train CCE")
    # plt.legend()
    # plt.show()
