from somlib import NeuralNet
import pandas as pd
import os

path = os.path.abspath(__file__).replace("test.py", "data/")

train_data = pd.read_csv(path + "train.csv")
valid_data = pd.read_csv(path + "test.csv")

settings = {
            "outs" : 5,
            "input_len" : len(train_data),
            "architecture" : [18],
            "inputs" : len(train_data.columns) - 5,
            "activation" : "sigmoid",
        }

# Построение ИНС
nn = NeuralNet(settings, verbose=True)

# Обучение ИНС (Метод Левенберга-Марквардта)
nn.fit_lm(
        x_train=train_data.values[:,:-5], 
        y_train=train_data.values[:,-5:],
        x_valid=valid_data.values[:,:-5],
        y_valid=valid_data.values[:,-5:],
        mu_init=5.0, 
        min_error=2.083e-4 , 
        max_steps=100, 
        mu_multiply=10, 
        mu_divide=10, 
        m_into_epoch=10, 
        verbose=True
    ) 

nn.plot_lw(None, save=False)