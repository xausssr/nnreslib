from pathlib import Path

import pandas as pd
from somlib import NeuralNet
from utils.hyperparametres import find_hyperparametres

path = Path(__file__).parent / "data"

train_data = pd.read_csv(path / "train.csv")
valid_data = pd.read_csv(path / "test.csv")

input_len = len(train_data)
output_len = len(train_data.columns) - 5

# Настройки гиперпараметров
settings_hyperparametres = find_hyperparametres(input_len, 5, output_len)

# Архитектура ИНС
architecture = {
    "l1": {"type": "fully_conneted", "neurons": settings_hyperparametres["neurons_mid"][0], "activation": "sigmoid"},
    "l2": {"type": "fully_conneted", "neurons": settings_hyperparametres["neurons_mid"][1], "activation": "tanh"},
    "out": {"type": "out", "neurons": 5, "activation": "softmax"},
}

# Настройки обучения
settings = {
    "outs": 5,
    "input_len": input_len,
    "architecture": architecture,
    "inputs": output_len,
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
    min_error=settings_hyperparametres["stop"],
    max_steps=100,
    mu_multiply=10,
    mu_divide=10,
    m_into_epoch=10,
    verbose=True,
)

# Отрисовка обучения ИНС
nn.plot_lw(None, save=False)

print(nn.predict(valid_data.values[:, :-5], raw=True)[:10])
print(nn.predict(valid_data.values[:, :-5], raw=False)[:10])
