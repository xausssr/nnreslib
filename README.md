# LevenbergLib

## Description

Library for research and teaching of neural networks using second-order methods

For testing lib: `python.exe test.py`

## Model compile
### Architecture
Architecture of neural network defining by `dict` object. 

Every `key` of this `dict` is name of layer, e.g. "input", "layer_1", "first layer", ect. This names choose by user. For each layer (`key`) `value` is another `dict`, vith keys:
* `type`: layer type, required key; now available: 
    * `"fully_connected"` - for fully-connected layer;
    * `"convolution"` - for convolution layer;
    * `"max_pool"` - for max pooling layer;
    * `"flatten"` - for reshape n-demetion tensors (for example: output of convolution layer) into vector (one-dimention object). For batch of objects reshape all objects in batch, but save first dimention <img src="https://render.githubusercontent.com/render/math?math=(10 \times 5 \times 3 \times 2) \rightarrow (10 \times 30)">
    * `"out"` - for last fully-connected layer (e.g. output of model)
* "`activation"` - non-lineary function for all layers, <u>except</u> **max pooling** and **flatten**, required key; now available: 
    * `"sigmoid"`: <p align="center"> <img src="https://render.githubusercontent.com/render/math?math=h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} %2B e^- \theta^Tx }"></p>
 
    * `"tanh"`:<p align="center"> <img src="https://render.githubusercontent.com/render/math?math=tanh(x) = \frac{e^{2x} - 1}{e^{2x} %2B 1}"></p>

    * `"relu"`:<p align="center"> <img src="https://render.githubusercontent.com/render/math?math=ReLU(x) = max(x, 0)"></p>

    * `"softmax"`:<p align="center"> <img src="https://render.githubusercontent.com/render/math?math=\sigma (x)_{i} = \frac{e^{x_{i}}}{\sum^{K}_{k=1}{e^{x_{k}}}}"></p>

* specific keys for **convolution layer**:
    * `"filtres"`: number of filters in convolution layer, dtype: `int`;
    * `"kernel"` : shape of filter (convolution kernel), this key recive list of 2 int for "width" and "heigh" of convolution kernel, dtype: `list`;
    * `"stride"` : stride along "width" and "heigh" for convolution operation, this key recive list of 2 int for "width" and "heigh" stride, dtype: `list`;
    * `"pad"` : padding input tensor with zeros along "width" and "heigh", this key recive list of 2 int for "width" and "heigh" stride, dtype: `list`;
* specific keys for **max pooling layer**:
    * `"kernel"` : shape of max pooling mask, this key recive list of 2 int for "width" and "heigh" of max pooling mask, dtype: `list`;
    * `"stride"` : stride along "width" and "heigh" for max pooling, this key recive list of 2 int for "width" and "heigh" stride, dtype: `list`;
* specific keys for **fully-connected layer**:
    * `"neurons"`: number of hidden units (neurons) into layer, dtype: `int`
* **flatten layer** has no parameters.

-----
## Описание

Библиотека для исследования и обучения нейронных сетей методами второго порядка

Для тестирования библиотеки: `python.exe test.py`

### Архитектура
Архитектура нейронной сети определяется объектом `dict` (словарь). 

Каждый ключ `key` данного словаря `dict` содержит имя слоя, например, "input", "layer_1", "first layer", и т.д. Имена определяется пользователем. Для каждого слоя (`key`) значение `value` - это вложенный словарь `dict`, с ключами:
* `type`: тип слоя нейронной сети, обязательный ключ; сейчас доступны: 
    * `"fully_connected"` - для полносвязного слоя;
    * `"convolution"` - для сверточного слоя;
    * `"max_pool"` - для слоя подвыборки;
    * `"flatten"` - для преобразования n-мерного тензора (например: выход сверточного слоя) в вектор (одномерный объект). Для работы в режиме мини-пакетов преобразуются все входные тензоры, но первое измерение сохраняется <img src="https://render.githubusercontent.com/render/math?math=(10 \times 5 \times 3 \times 2) \rightarrow (10 \times 30)">
    * `"out"` - последный полносвязный слой (выход модели)
* "`activation"` - нелинейные функции для всех слоев, <u>за исключением</u> **max_pool** and **flatten**, обязательный ключ; сейчас доступны: 
    * `"sigmoid"`:<p align="center"> <img src="https://render.githubusercontent.com/render/math?math=h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} %2B e^- \theta^Tx }"></p>
 
    * `"tanh"`:<p align="center"> <img src="https://render.githubusercontent.com/render/math?math=tanh(x) = \frac{e^{2x} - 1}{e^{2x} %2B 1}"></p>

    * `"relu"`: <p align="center"> <img src="https://render.githubusercontent.com/render/math?math=ReLU(x) = max(x, 0)"></p>

    * `"softmax"`:<p align="center"> <img src="https://render.githubusercontent.com/render/math?math=\sigma (x)_{i} = \frac{e^{x_{i}}}{\sum^{K}_{k=1}{e^{x_{k}}}}"></p>

Example of architecture dict:
```python
architecture = {
    "input": {"type": "fully_conneted", "neurons": 31, "activation": "sigmoid"},
    "hidden": {"type": "fully_conneted", "neurons": 18, "activation": "sigmoid"},
    "out": {"type": "out", "neurons": 5, "activation": "sigmoid"},
}
```

* специфичные ключи для **сверточного слоя**:
    * `"filtres"`: количество фильтров в сверточном слое, dtype: `int`;
    * `"kernel"` : размерность фильтра (сверточного ядра), данный ключ должен содержать список `list` длинной 2 с целочисленными данными `int` отображающими "ширину" и "высоту" сверточного ядра, dtype: `list`;
    * `"stride"` : смещение вдоль "ширины" и "высоты" для операции свертки, данный ключ должен содержать список `list` длинной 2 с целочисленными данными `int` отображающими "смещение по ширине" и "смещение по высоте", dtype: `list`;
    * `"pad"` : заполнение нулями (дополнительная граница из нулей) входного тензора вдоль "ширины" и "высоты", данный ключ должен содержать список `list` длинной 2 с целочисленными данными `int` отображающими "заполнение по ширине" и "заполнение по высоте", dtype: `list`;
* специфичные ключи для **слоя подвыборки**:
    * `"kernel"` : размерность фильтра (сверточного ядра), данный ключ должен содержать список `list` длинной 2 с целочисленными данными `int` отображающими "ширину" и "высоту" сверточного ядра, dtype: `list`;
    * `"stride"` : смещение вдоль "ширины" и "высоты" для операции свертки, данный ключ должен содержать список `list` длинной 2 с целочисленными данными `int` отображающими "смещение по ширине" и "смещение по высоте", dtype: `list`;
* специфичные ключи для **полносвязного слоя**:
    * `"neurons"`: количество нейронов в слое, dtype: `int`
* слой **flatten** не имеет настраиваемых параметров.

Пример словоря с описанием архитектуры:
```python
architecture = {
    "input": {"type": "fully_conneted", "neurons": 31, "activation": "sigmoid"},
    "hidden": {"type": "fully_conneted", "neurons": 18, "activation": "sigmoid"},
    "out": {"type": "out", "neurons": 5, "activation": "sigmoid"},
}
```

