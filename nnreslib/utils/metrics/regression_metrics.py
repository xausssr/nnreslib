from typing import Callable, Dict, Sequence

import numpy as np


def __mse(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray], sqrt_func: Callable[[float], float]) -> float:
    result = 0.0
    for out_true, out_pred in zip(vec_true, vec_pred):
        result += sqrt_func(np.mean((out_true - out_pred) ** 2))
    return result / len(vec_true)


def _mse(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> float:
    return __mse(vec_true, vec_pred, lambda x: x)


def _rmse(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> float:
    return __mse(vec_true, vec_pred, np.sqrt)


def _mae(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> float:
    result = 0.0
    for out_true, out_pred in zip(vec_true, vec_pred):
        result += np.mean(np.abs(out_true - out_pred))
    return result / len(vec_true)


MetricType = Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], float]

REGRESSION_METRICS: Dict[str, MetricType] = dict(
    MSE=_mse,
    RMSE=_rmse,
    MAE=_mae,
)
