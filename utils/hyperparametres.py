import math
import numpy as np
from enum import Enum, auto
from typing import List, Tuple, Dict

class BorderAssessment(Enum):
    LOW = auto()
    MID = auto()
    UP = auto()

class HiddenLayersCountException(ValueError):
    """
    Exeption raised for count hidden layers
    """
    pass

MIN_NUM_HIDDEN_LAYERS = 1
MAX_NUM_HIDDEN_LAYERS = 2

def det_stop(
    len_dataset: int,
    len_output: int,
    gamma: float = 0.5
    ) -> Tuple[float, float]:
    """
    Deterministic method for finding the learning stop criterion
    len_dataset : int - training sample size
    len_output : int - number of neurons in the output layer
    gamma : float - error threshold
    
    e_h_s_lg - stopping criterion in decibels
    """
    e_n_0 = 0.25
    e_n_s = 1 / (2 * len_dataset * len_output) * np.max([gamma ** 2, (1 - gamma) ** 2])
    e_h_s = e_n_s / e_n_0
    e_h_s_lg = - 3 - 10 * np.log10(len_dataset) - 10 * np.log10(len_output)
    return e_h_s, e_h_s_lg

def count_epochs(
    len_input: int,
    len_output: int,
    len_dataset: int,
    e_h_s_lg: float,
    neurons: List[int]
    ) -> int:
    """
    Calculates the number of epoch for training
    len_dataset : int - training sample size
    len_output : int - number of neurons in the output layer
    len_input : int - number of input parameters
    e_h_s_lg : float - stopping criterion in decibels
    neurons : list, tuple - distribution of neurons to hidden layers
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

def num_neurons(
    len_input: int,
    len_output: int,
    len_dataset: int,
    num_hidden_layers: int = 2,
    border_assessment: BorderAssessment = BorderAssessment.MID
    ) -> List[int]:
    """
    len_input - number of input parametres
    len_output - number of neurons in the output layer
    len_dataset - training sample size
    num_hidden_layers - number of hidden layers
    border_assessment : BorderAssessment
    """
    sum_parametres = len_input + len_output
    w_low = len_output * len_dataset / (1 + math.log2(len_dataset))
    w_up = len_output * (len_dataset / len_input + 1) * (sum_parametres + 1) + len_output

    assert MIN_NUM_HIDDEN_LAYERS <= num_hidden_layers <= MAX_NUM_HIDDEN_LAYERS

    if border_assessment == BorderAssessment.LOW:
        n_value = w_low / sum_parametres
    elif border_assessment == BorderAssessment.MID:
        w_mid = (w_low + w_up) / 2
        n_value = w_mid / sum_parametres
    elif border_assessment == BorderAssessment.UP:
        n_value = w_up / sum_parametres

    return {
        1:[math.ceil(n_value)],
        2:[math.ceil(2 / 3 * n_value + 1), math.ceil(1 / 3 * n_value + 1)]
    }[num_hidden_layers]
    
def find_hyperparametres(
    len_dataset: int,
    len_output: int,
    len_input: int,
    num_hidden_layers: int = 2
    ) -> Tuple[Dict[BorderAssessment, Dict[str, List[int]]], Tuple[float, float]]:

    if (MIN_NUM_HIDDEN_LAYERS > num_hidden_layers) or (MAX_NUM_HIDDEN_LAYERS < num_hidden_layers):
        raise HiddenLayersCountException(f'Number hidden layers is not correct: min = {MIN_NUM_HIDDEN_LAYERS}, max = {MAX_NUM_HIDDEN_LAYERS}')

    stop = det_stop(len_dataset, len_output)
    dict_hyper: Dict[BorderAssessment, Dict[str, List[int]]] = {}

    for border_assesment in BorderAssessment:
        neurons = num_neurons(len_input, len_output, len_dataset, num_hidden_layers, border_assesment)
        epochs = count_epochs(len_input, len_output, len_dataset, stop[1], neurons)
        dict_hyper[border_assesment] = dict(num_neurons=neurons, num_epochs=epochs)
    return dict_hyper, stop
