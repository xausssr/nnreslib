from typing import Callable, Dict, List, Sequence

import numpy as np


def _mse(
    vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray], sqrt_func: Callable[[float], float]
) -> List[float]:
    return [sqrt_func(np.mean((out_true - out_pred) ** 2)) for out_true, out_pred in zip(vec_true, vec_pred)]


def mse(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> List[float]:
    return _mse(vec_true, vec_pred, lambda x: x)


def rmse(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> List[float]:
    return _mse(vec_true, vec_pred, np.sqrt)


def mae(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> List[float]:
    return [np.mean(np.abs(out_true - out_pred)) for out_true, out_pred in zip(vec_true, vec_pred)]


MetricType = Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], List[float]]

REGRESSION_METRICS: Dict[str, MetricType] = dict(
    MSE=mse,
    RMSE=rmse,
    MAE=mae,
)
