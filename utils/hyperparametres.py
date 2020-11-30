import math
import numpy as np
from enum import Enum, auto
from typing import List, Tuple, Dict

MIN_NUM_HIDDEN_LAYERS = 1
MAX_NUM_HIDDEN_LAYERS = 2

class BorderAssessment(Enum):
    LOW = auto()
    MID = auto()
    UP = auto()

class HiddenLayersCountException(ValueError):
    """
    Exeption raised for count hidden layers
    """

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
    
    stop_criterion_decibels - stopping criterion in decibels
    """
    error_norm_0_epoch = 0.25
    error_norm_s_epoch = 1 / (2 * len_dataset * len_output) * np.max([gamma ** 2, (1 - gamma) ** 2])
    stop_criterion = error_norm_s_epoch / error_norm_0_epoch
    stop_criterion_decibels = - 3 - 10 * np.log10(len_dataset) - 10 * np.log10(len_output)
    return stop_criterion, stop_criterion_decibels

def count_epochs(
    len_input: int,
    len_output: int,
    len_dataset: int,
    stop_criterion_decibels: float,
    neurons: List[int]
    ) -> int:
    """
    Calculates the number of epoch for training
    len_dataset : int - training sample size
    len_output : int - number of neurons in the output layer
    len_input : int - number of input parameters
    stop_criterion_decibels : float - stopping criterion in decibels
    neurons : list, tuple - distribution of neurons to hidden layers
    """

    if stop_criterion_decibels > 0:
        raise ValueError('stop_criterion_decibels must be negative')

    s_value = len_dataset * np.sqrt(-0.5 * stop_criterion_decibels)
    weight_connections = (len_input + 1) * neurons[0]
    for i in range(len(neurons) - 1):
        weight_connections += (neurons[i] + 1) * neurons[i+1]
    weight_connections += (neurons[-1] + 1) * len_output
    wieght_norm = weight_connections / (len_input * len_output)
    num_layers = np.sum(neurons) + len_output + len(neurons) + 1
    num_weight_coeff = weight_connections / num_layers
    s_value_all = s_value / (wieght_norm * num_weight_coeff) ** (1 / 4)
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
    weight_low = len_output * len_dataset / (1 + math.log2(len_dataset))
    weight_up = len_output * (len_dataset / len_input + 1) * (sum_parametres + 1) + len_output

    assert MIN_NUM_HIDDEN_LAYERS <= num_hidden_layers <= MAX_NUM_HIDDEN_LAYERS

    if border_assessment == BorderAssessment.LOW:
        n_value = weight_low / sum_parametres
    elif border_assessment == BorderAssessment.MID:
        weight_mid = (weight_low + weight_up) / 2
        n_value = weight_mid / sum_parametres
    elif border_assessment == BorderAssessment.UP:
        n_value = weight_up / sum_parametres

    if num_hidden_layers == 1:
        return [math.ceil(n_value)]
    elif num_hidden_layers == 2:
        return [math.ceil(2 / 3 * n_value + 1), math.ceil(1 / 3 * n_value + 1)]
    else: raise HiddenLayersCountException('Available 1 and 2 hidden layers')
    
def find_hyperparametres(
    len_dataset: int,
    len_output: int,
    len_input: int,
    num_hidden_layers: int = 2
    ) -> Tuple[Dict[BorderAssessment, Dict[str, List[int]]], Tuple[float, float]]:

    if (num_hidden_layers < MIN_NUM_HIDDEN_LAYERS) or (MAX_NUM_HIDDEN_LAYERS < num_hidden_layers):
        raise HiddenLayersCountException(f'Number hidden layers is not correct: min = {MIN_NUM_HIDDEN_LAYERS}, max = {MAX_NUM_HIDDEN_LAYERS}')

    stop = det_stop(len_dataset, len_output)
    dict_hyper: Dict[BorderAssessment, Dict[str, List[int]]] = {}

    for border_assesment in BorderAssessment:
        neurons = num_neurons(len_input, len_output, len_dataset, num_hidden_layers, border_assesment)
        epochs = count_epochs(len_input, len_output, len_dataset, stop[1], neurons)
        dict_hyper[border_assesment] = dict(num_neurons=neurons, num_epochs=epochs)
    return dict_hyper, stop
