# import matplotlib.pyplot as plt
# from nnreslib.utils.metrics import OpMode

import numpy as np
import tensorflow as tf

from nnreslib.architecture import ArchitectureType
from nnreslib.layers import FullyConnectedLayer, InputLayer
from nnreslib.model import Model
from nnreslib.utils.types import ActivationFunctions, Shape


def test_iris_net_lm():
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)

    data = np.load("./tests/data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        FullyConnectedLayer("fc_2", neurons=6),
        FullyConnectedLayer(
            "fc_3",
            neurons=3,
            activation=ActivationFunctions.SOFTMAX,
            is_out=True,
        ),
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "LevenbergMarquardt",
        x_train,
        y_train,
        x_validation,
        y_validation,
        200,
        0.05,
        step_into_epoch=10,
        regularisation_factor_init=5.0,
        regularisation_factor_decay=10.0,
        regularisation_factor_increase=10.0,
        percent_random=0.2,
    )

    assert epoch < 200
    assert loss < 0.05


def test_iris_net_adam():
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)

    data = np.load("./tests/data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        FullyConnectedLayer("fc_2", neurons=6),
        FullyConnectedLayer(
            "fc_3",
            neurons=3,
            activation=ActivationFunctions.SOFTMAX,
            is_out=True,
        ),
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "Adam", x_train, y_train, x_validation, y_validation, 200, 0.1, learning_rate=0.01, logging_step=10
    )

    assert epoch <= 200
    assert loss < 0.1


def test_iris_net_adadelta():
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)

    data = np.load("./tests/data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        FullyConnectedLayer("fc_2", neurons=6),
        FullyConnectedLayer(
            "fc_3",
            neurons=3,
            activation=ActivationFunctions.SOFTMAX,
            is_out=True,
        ),
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "Adadelta",
        x_train,
        y_train,
        x_validation,
        y_validation,
        200,
        0.1,
        learning_rate=30.0,
        logging_step=10,
    )

    assert epoch <= 200
    assert loss < 0.1


def test_iris_net_adagrad():
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)

    data = np.load("./tests/data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        FullyConnectedLayer("fc_2", neurons=6),
        FullyConnectedLayer(
            "fc_3",
            neurons=3,
            activation=ActivationFunctions.SOFTMAX,
            is_out=True,
        ),
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "Adagrad",
        x_train,
        y_train,
        x_validation,
        y_validation,
        200,
        0.1,
        learning_rate=1.0,
        logging_step=10,
    )

    assert epoch <= 200
    assert loss < 0.1


def test_iris_net_rmsprop():
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)

    data = np.load("./tests/data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        FullyConnectedLayer("fc_2", neurons=6),
        FullyConnectedLayer(
            "fc_3",
            neurons=3,
            activation=ActivationFunctions.SOFTMAX,
            is_out=True,
        ),
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "RMSProp",
        x_train,
        y_train,
        x_validation,
        y_validation,
        200,
        0.1,
        learning_rate=0.1,
        logging_step=10,
    )

    assert epoch <= 200
    assert loss < 0.1


def test_iris_net_momentum():
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)

    data = np.load("./tests/data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        FullyConnectedLayer("fc_2", neurons=6),
        FullyConnectedLayer(
            "fc_3",
            neurons=3,
            activation=ActivationFunctions.SOFTMAX,
            is_out=True,
        ),
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "Momentum",
        x_train,
        y_train,
        x_validation,
        y_validation,
        200,
        0.1,
        learning_rate=5.0,
        logging_step=10,
    )

    assert epoch <= 200
    assert loss < 0.1


def test_iris_net_sgd():
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)

    data = np.load("./tests/data/iris.npy")
    np.random.shuffle(data)
    x_train = data[:150, :-1]
    y_train = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    x_validation = data[:150, :-1]
    y_validation = np.eye(3)[data[:150, -1].reshape((-1)).astype(int)].astype(np.float64)

    architecture: ArchitectureType = [
        InputLayer("input", Shape(4)),
        FullyConnectedLayer("fc_1", neurons=5),
        FullyConnectedLayer("fc_2", neurons=6),
        FullyConnectedLayer(
            "fc_3",
            neurons=3,
            activation=ActivationFunctions.SOFTMAX,
            is_out=True,
        ),
    ]

    model = Model(150, architecture)
    epoch, loss = model.train(
        "SGD",
        x_train,
        y_train,
        x_validation,
        y_validation,
        200,
        0.1,
        learning_rate=5.0,
        logging_step=10,
    )

    assert epoch <= 200
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
