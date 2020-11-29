from pathlib import Path

import pandas as pd
import numpy as np
from somlib import NeuralNet

path = Path(__file__).parent / "data"

np.set_printoptions(precision=3)

print("\n\nТест сверточной сети на наборе MNIST")

train_data = pd.read_csv(path / 'mnist_train.csv', delimiter=',').values[:500]
valid_data = pd.read_csv(path / 'mnist_test.csv', delimiter=',').values[:500]

architecture = {
    "cl1": {"type": "convolution", "filtres": 8, "kernel" : [3, 3], "stride": [2, 2], "pad": [0, 0], "activation" : "relu"},
    "mp1": {"type": "max_pool", "kernel" : [2, 2], "stride": [2, 2]},
    "cl2": {"type": "convolution", "filtres": 8, "kernel" : [3, 3], "stride": [2, 2], "pad": [0, 0], "activation" : "relu"},
    "mp2": {"type": "max_pool", "kernel" : [2, 2], "stride": [2, 2]},
    "fl": {"type": "flatten"},
    "l1": {"type": "fully_conneted", "neurons": 28, "activation": "sigmoid"},
    "l2": {"type": "fully_conneted", "neurons": 12, "activation": "tanh"},
    "out": {"type": "out", "neurons": 10, "activation": "softmax"},
}

settings = {
    "outs": 10,
    "batch_size": 100,
    "architecture": architecture,
    "inputs": [28,28,1],
    "activation": "sigmoid",
}

# Построение CНС
nn = NeuralNet(settings, verbose=True)

nn.fit_lm(
    x_train=train_data[:,1:].reshape((-1,28,28,1)),
    y_train=np.eye(10)[train_data[:, 0]],
    x_valid=valid_data[:,1:].reshape((-1,28,28,1)),
    y_valid=np.eye(10)[valid_data[:, 0]],
    mu_init=5.0,
    min_error=2.083e-4,
    max_steps=6,
    mu_multiply=5,
    mu_divide=5,
    m_into_epoch=5,
    verbose=True,
    random_batches=True
)

# Отрисовка обучения ИНС
nn.plot_lw(None, save=False, logscale=False)

print(nn.predict(valid_data[:,1:].reshape((-1,28,28,1)), raw=True)[:5])
print(nn.predict(valid_data[:,1:].reshape((-1,28,28,1)), raw=False)[:5])

print("\n\nТест полносвязной сети")

train_data = pd.read_csv(path / "train.csv")
valid_data = pd.read_csv(path / "test.csv")

# Архитектура ИНС
architecture = {
    "l1": {"type": "fully_conneted", "neurons": 31, "activation": "sigmoid"},
    "l2": {"type": "fully_conneted", "neurons": 18, "activation": "sigmoid"},
    "out": {"type": "out", "neurons": 5, "activation": "sigmoid"},
}

# Настройки обучения
settings = {
    "outs": 5,
    "batch_size": 110,
    "architecture": architecture,
    "inputs": [len(train_data.columns) - 5],
    "activation": "sigmoid",
}

# Построение ИНС
nn = NeuralNet(settings, verbose=True)

# Обучение ИНС (Метод Левенберга-Марквардта)
nn.fit_lm(
    x_train=train_data.values[:, :-5],
    y_train=train_data.values[:, -5:],
    x_valid=valid_data.values[:, :-5],
    y_valid=valid_data.values[:, -5:],
    mu_init=5.0,
    min_error=2.083e-4,
    max_steps=1000,
    mu_multiply=10,
    mu_divide=10,
    m_into_epoch=5,
    verbose=True,
    random_batches=True
)

# Отрисовка обучения ИНС
nn.plot_lw(None, save=False, logscale=False)

print(nn.predict(valid_data.values[:, :-5], raw=True)[:10])
print(nn.predict(valid_data.values[:, :-5], raw=False)[:10])
