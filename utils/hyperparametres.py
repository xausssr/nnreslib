import math
import numpy as np


def det_stop(Q, P, gamma=0.5):
    """
    Детерминированный способ нахождения критерия останова обучения
    Q : int - объем обучающей выборки
    P : int - кол-во нейронов в выходном слое
    gamma : float - порог ошибок решений
    
    E_H_S_lg - критерий остановки в децибелах
    """
    E_N_0 = 0.25
    E_N_S = 1/(2*Q*P)*np.max([gamma**2, (1-gamma)**2])
    E_H_S = E_N_S/E_N_0
    E_H_S_lg = -3-10*np.log10(Q)-10*np.log10(P)
    return E_H_S, E_H_S_lg

def count_epochs(Q, P, M, E_H_S_lg, neurons):
    """
    Вычисляет кол-во эпох для обучения
    Q : int - объем обучающей выборки
    P : int - кол-во нейронов в выходном слое
    M : int - кол-во входных параметров
    E_H_S_lg : float - критерий остановки в децибелах
    neurons : list, tuple - распределение нейронов по скрытым слоям   
    """

    if E_H_S_lg > 0:
        raise ValueError('E_H_S_lg must be negative')

    s = Q*np.sqrt(-0.5*E_H_S_lg)
    W_E = (M + 1)*neurons[0] 
    for i in range(len(neurons)-1):
        W_E += (neurons[i] + 1)*neurons[i+1]
    W_E += (neurons[-1] + 1)*P
    W_MP = W_E/(M*P)
    N_E = np.sum(neurons) + P + len(neurons) + 1
    w_N = W_E/N_E
    S = s/(W_MP*w_N)**(1/4)
    S += S*0.1 
    return math.ceil(S)

def num_neurons(m, p, q, h=2, border_assessment="up"):    
    """
    m - кол-во входов НС(число входных параметров)
    p - кол-во нейронов в выходном слое НС
    q - объем обучающей выборки
    h - количество скрытых слоев
    border_assessment : "low", "mid", "up" 
    """

    w_low = p*q/(1 + math.log2(q))  # w - синаптические веса
    w_up = p*(q/m + 1)*(m + p + 1) + p
    w_mid = (w_low + w_up)/2

    if border_assessment == "low":
        n = w_low/(m + p)
    elif border_assessment == "mid":
        n = w_mid/(m + p)
    elif border_assessment == "up":
        n = w_up/(m + p)

    if h == 1:
        return math.ceil(n)
    if h == 2:
        n1 = math.ceil(2/3*n + 1)
        n2 = math.ceil(1/3*n + 1)
        return n1, n2
    
def find_hyperparametres(Q, P, M, h=2):
	stop = det_stop(Q, P)
	neurons_low = num_neurons(M, P, Q, h, border_assessment='low')
	neurons_mid = num_neurons(M, P, Q, h, border_assessment='mid')
	neurons_up = num_neurons(M, P, Q, h, border_assessment='up')
	epochs_low = count_epochs(Q, P, M, stop[1], neurons_low)
	epochs_mid = count_epochs(Q, P, M, stop[1], neurons_mid)
	epochs_up = count_epochs(Q, P, M, stop[1], neurons_up)

	dict_hyperparam = {'stop':stop[0],
		        'stop_dB':stop[1],
		        'neurons_low':neurons_low,
		        'neurons_mid':neurons_mid,
		        'neurons_up':neurons_up,
		        'epochs_low':epochs_low,
		        'epochs_mid':epochs_mid,
		        'epochs_up':epochs_up}

	return dict_hyperparam

