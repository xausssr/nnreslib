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
 * `"sigmoid"`:
 <p align="center"> <img src="https://render.githubusercontent.com/render/math?math=h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} %2B e^- \theta^Tx }"></p>
 
 * `"tanh"`:
 <p align="center"> <img src="https://render.githubusercontent.com/render/math?math=tanh(x) = \frac{e^{2x} - 1}{e^{2x} %2B 1}"></p>

 * `"relu"`:
 <p align="center"> <img src="https://render.githubusercontent.com/render/math?math=ReLU(x) = max(x, 0)"></p>
 * `"softmax"`:
<p align="center"> <img src="https://render.githubusercontent.com/render/math?math=\sigma (x)_{i} = \frac{e^{x_{i}}}{\sum^{K}_{k=1}{e^{x_{k}}}}"></p>

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
## Описание

Библиотека для исследования и обучения нейронных сетей методами второго порядка

Для тестирования библиотеки: `python.exe test.py`

Список задач:

1. Отформатировать данное readme (добавить markdown-разметку)
2. ~~Переписать логику построения архитектуры ИНС -- сделать более детальной~~
3. Добавить поддержку сверточных нейронных сетей
4. Добавить больше целевых функций ошибки в граф вычислений (tf)
5. Добавить возможность изменять критерий останова через настройки
6. Добавить больше метрик для истории
