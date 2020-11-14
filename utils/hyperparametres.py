import math
import numpy as np
from typing import List, Tuple

def det_stop(len_dataset: int, len_output: int, gamma: float = 0.5) -> Tuple[float, float]:
    """
    Детерминированный способ нахождения критерия останова обучения
    len_dataset : int - объем обучающей выборки
    len_output : int - кол-во нейронов в выходном слое
    gamma : float - порог ошибок решений
    
    e_h_s_lg - критерий остановки в децибелах
    """
    e_n_0 = 0.25
    e_n_s = 1 / (2 * len_dataset * len_output) * np.max([gamma ** 2, (1 - gamma) ** 2])
    e_h_s = e_n_s / e_n_0
    e_h_s_lg = - 3 - 10 * np.log10(len_dataset) - 10 * np.log10(len_output)
    return e_h_s, e_h_s_lg

def count_epochs(len_dataset: int, len_output: int, len_input: int, e_h_s_lg: float, neurons: List[int]) -> int:
    """
    Вычисляет кол-во эпох для обучения
    len_dataset : int - объем обучающей выборки
    len_output : int - кол-во нейронов в выходном слое
    len_input : int - кол-во входных параметров
    e_h_s_lg : float - критерий остановки в децибелах
    neurons : list, tuple - распределение нейронов по скрытым слоям   
    """

    if e_h_s_lg > 0:
        raise ValueError('e_h_s_lg must be negative')

    s_value = len_dataset * np.sqrt(-0.5 * e_h_s_lg)
    w_e = (len_input + 1) * neurons[0] 
    for i in range(len(neurons) - 1):
        w_e += (neurons[i] + 1) * neurons[i+1]
    w_e += (neurons[-1] + 1) * len_output
    w_mp = w_e / (len_input * len_output)
    n_e = np.sum(neurons) + len_output + len(neurons) + 1
    w_n = w_e / n_e
    s_value_all = s_value / (w_mp * w_n) ** (1 / 4)
    s_value_all += s_value_all * 0.1 
    return math.ceil(s_value_all)

def num_neurons(len_input: int, len_output: int, len_dataset: int, num_hidden_layers: int = 2, border_assessment: str = "up") -> List[int]:    
    """
    len_input - кол-во входов НС(число входных параметров)
    len_output - кол-во нейронов в выходном слое НС
    len_dataset - объем обучающей выборки
    num_hidden_layers - количество скрытых слоев
    border_assessment : "low", "mid", "up" 
    """

    w_low = len_output * len_dataset/(1 + math.log2(len_dataset))  # w - синаптические веса
    w_up = len_output * (len_dataset/len_input + 1) * (len_input + len_output + 1) + len_output
    w_mid = (w_low + w_up) / 2

    if border_assessment == "low":
        n_value = w_low / (len_input + len_output)
    elif border_assessment == "mid":
        n_value = w_mid / (len_input + len_output)
    elif border_assessment == "up":
        n_value = w_up / (len_input + len_output)

    if num_hidden_layers == 1:
        return math.ceil(n_value)
    if num_hidden_layers == 2:
        n1 = math.ceil(2 / 3 * n_value + 1)
        n2 = math.ceil(1 / 3 * n_value + 1)
        return n1, n2
    
def find_hyperparametres(len_dataset: int, len_output: int, len_input: int, num_hidden_layers: int = 2) -> dict:
	stop = det_stop(len_dataset, len_output)
	neurons_low = num_neurons(len_input, len_output, len_dataset, num_hidden_layers, border_assessment='low')
	neurons_mid = num_neurons(len_input, len_output, len_dataset, num_hidden_layers, border_assessment='mid')
	neurons_up = num_neurons(len_input, len_output, len_dataset, num_hidden_layers, border_assessment='up')
	epochs_low = count_epochs(len_dataset, len_output, len_input, stop[1], neurons_low)
	epochs_mid = count_epochs(len_dataset, len_output, len_input, stop[1], neurons_mid)
	epochs_up = count_epochs(len_dataset, len_output, len_input, stop[1], neurons_up)

	dict_hyperparam = {'stop':stop[0],
		        'stop_dB':stop[1],
		        'neurons_low':neurons_low,
		        'neurons_mid':neurons_mid,
		        'neurons_up':neurons_up,
		        'epochs_low':epochs_low,
		        'epochs_mid':epochs_mid,
		        'epochs_up':epochs_up}

	return dict_hyperparam

