from pathlib import Path

import pandas as pd
from somlib import NeuralNet

path = Path(__file__).parent / "data"

print("\n\nТест сверточной сети на наборе MNIST")

architecture = {
    "cl1": {"type": "convolution", "filtres": 8, "kernel" : [3, 3], "stride": [2, 2], "pad": [0, 0]},
    "mp1": {"type": "max_pool", "kernel" : [2, 2], "stride": [2, 2]},
    "cl2": {"type": "convolution", "filtres": 8, "kernel" : [3, 3], "stride": [2, 2], "pad": [0, 0]},
    "mp2": {"type": "max_pool", "kernel" : [2, 2], "stride": [2, 2]},
    "fl": {"type": "flatten"},
    "l1": {"type": "fully_conneted", "neurons": 28, "activation": "sigmoid"},
    "l2": {"type": "fully_conneted", "neurons": 12, "activation": "tanh"},
    "out": {"type": "out", "neurons": 5, "activation": "softmax"},
}

settings = {
    "outs": 10,
    "batch_size": 120,
    "architecture": architecture,
    "inputs": [28,28,1],
    "activation": "sigmoid",
}

# Построение CНС
nn = NeuralNet(settings, verbose=True)
exit()

train_data = pd.read_csv(path / "train.csv")
valid_data = pd.read_csv(path / "test.csv")

# Архитектура ИНС
architecture = {
    "l1": {"type": "fully_conneted", "neurons": 28, "activation": "sigmoid"},
    "l2": {"type": "fully_conneted", "neurons": 12, "activation": "tanh"},
    "out": {"type": "out", "neurons": 5, "activation": "softmax"},
}

# Настройки обучения
settings = {
    "outs": 5,
    "batch_size": 120,
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
    max_steps=6,
    mu_multiply=5,
    mu_divide=5,
    m_into_epoch=5,
    verbose=True,
)

# Отрисовка обучения ИНС
nn.plot_lw(None, save=False, logscale=False)

print(nn.predict(valid_data.values[:, :-5], raw=True)[:10])
print(nn.predict(valid_data.values[:, :-5], raw=False)[:10])
