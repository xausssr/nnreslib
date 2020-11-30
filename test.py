from pathlib import Path

import pandas as pd
import numpy as np
from somlib import NeuralNet
from utils.hyperparametres import find_hyperparametres, BorderAssessment

path = Path(__file__).parent / "data"

np.set_printoptions(precision=3)

print("\n\nTest convolution architecture on MNIST dataset")

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

# build CNN
nn = NeuralNet(settings, verbose=True)

nn.fit_lm(
    x_train=train_data[:,1:].reshape((-1,28,28,1)),
    y_train=np.eye(10)[train_data[:, 0]],
    x_valid=valid_data[:,1:].reshape((-1,28,28,1)),
    y_valid=np.eye(10)[valid_data[:, 0]],
    mu_init=5.0,
    min_error=2.083e-4,
    max_steps=3,
    mu_multiply=5,
    mu_divide=5,
    m_into_epoch=5,
    verbose=True,
    random_batches=True
)

# Plot errors CNN
nn.plot_lw(None, save=False, logscale=False)

print(nn.predict(valid_data[:,1:].reshape((-1,28,28,1)), raw=True)[:5])
print(nn.predict(valid_data[:,1:].reshape((-1,28,28,1)), raw=False)[:5])

print("\n\nPerceptron test")

train_data = pd.read_csv(path / "train.csv")
valid_data = pd.read_csv(path / "test.csv")

len_dataset = len(train_data)
input_len = len(train_data.columns) - 5

# Settings hyperparametres
settings_hyperparametres, stop = find_hyperparametres(len_dataset, 5, input_len)

# Build perceptron
architecture = {
    "l1": {"type": "fully_conneted", "neurons": settings_hyperparametres[BorderAssessment.MID]["num_neurons"][0], "activation": "sigmoid"},
    "l2": {"type": "fully_conneted", "neurons": settings_hyperparametres[BorderAssessment.MID]["num_neurons"][1], "activation": "tanh"},
    "out": {"type": "out", "neurons": 5, "activation": "softmax"},
}

# Settings of train
settings = {
    "outs": 5,
    "batch_size": 120,
    "architecture": architecture,
    "inputs": [input_len],
    "activation": "sigmoid",
}

nn = NeuralNet(settings, verbose=True)

# Train NN (Levenberg-Marqvardt method)
nn.fit_lm(
    x_train=train_data.values[:, :-5],
    y_train=train_data.values[:, -5:],
    x_valid=valid_data.values[:, :-5],
    y_valid=valid_data.values[:, -5:],
    mu_init=5.0,
    min_error=stop[0],
    max_steps=100,
    mu_multiply=5,
    mu_divide=5,
    m_into_epoch=5,
    verbose=True,
    random_batches=True
)

# Plot erros NN
nn.plot_lw(None, save=False, logscale=False)

print(nn.predict(valid_data.values[:, :-5], raw=True)[:10])
print(nn.predict(valid_data.values[:, :-5], raw=False)[:10])
