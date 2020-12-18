from __future__ import annotations

import collections.abc as ca
from collections import defaultdict
from enum import Enum, auto
from typing import Callable, Dict, List

import numpy as np

from . import log

logger = log.get(__name__)


class MetricResult:
    def __init__(self, out_values: np.ndarray) -> None:
        self.out_values = out_values
        self.mean_value = np.mean(out_values)

    def __add__(self, metric_result: MetricResult) -> MetricResult:
        return type(self)(self.out_values + metric_result.out_values)

    def __eq__(self, metric_result: object) -> bool:
        if isinstance(metric_result, type(self)):
            return np.array_equal(self.out_values, metric_result.out_values)
        return False

    def __lt__(self, metric_result: object) -> bool:
        if isinstance(metric_result, type(self)):
            return np.prod(np.less(self.out_values, metric_result.out_values)) == 1  # type: ignore
        raise TypeError(
            f"'<' not supported between instance of '{type(self).__name__}' and '{type(metric_result).__name__}'"
        )

    def __truediv__(self, value: object) -> MetricResult:
        if isinstance(value, (int, float)):
            return type(self)(self.out_values / value)
        raise TypeError(f"unsupported operand type(s) for /: '{type(self).__name__}' and '{type(value).__name__}'")


_MetricType = Callable[[np.ndarray, np.ndarray], np.ndarray]
MetricType = Callable[[np.ndarray, np.ndarray], MetricResult]


def metric_adapter(func: _MetricType) -> MetricType:
    def adapter(vec_true: np.ndarray, vec_pred: np.ndarray) -> MetricResult:
        return MetricResult(func(vec_true, vec_pred))

    return adapter


@metric_adapter
def _mse(vec_true: np.ndarray, vec_pred: np.ndarray) -> np.ndarray:
    return np.sum(np.sqrt(np.power(vec_true - vec_pred, 2)), axis=0)


@metric_adapter
def _mae(vec_true: np.ndarray, vec_pred: np.ndarray) -> np.ndarray:
    if len(vec_true.shape) == 2 and len(vec_pred.shape) == 2:
        c_vec_true = np.argmax(vec_true)
        c_vec_pred = np.argmax(vec_pred)
        return np.array([np.sum(np.abs(c_vec_true - c_vec_pred))])
    logger.warning("Metric MAE is categorical! You need use 2D one-hot encoded arrays for output result")
    return np.zeros(1)


@metric_adapter
def _cce(vec_true: np.ndarray, vec_pred: np.ndarray) -> np.ndarray:
    if len(vec_true.shape) == 2 and len(vec_pred.shape) == 2:
        return np.array([np.sum(-(vec_true * np.log(vec_pred)))])
    logger.warning(
        "Metric Categorical cross-entropy is categorical! You need use 2D one-hot encoded arrays for output result"
    )
    return np.zeros(1)


@metric_adapter
def _roc(vec_true: np.ndarray, vec_pred: np.ndarray) -> np.ndarray:
    raise NotImplementedError()


@metric_adapter
def _auc(vec_true: np.ndarray, vec_pred: np.ndarray) -> np.ndarray:
    raise NotImplementedError()


STANDART_METRICS: Dict[str, MetricType] = dict(
    MSE=_mse,
    MAE=_mae,
    CC=_cce,
    ROC=_roc,
    AUC=_auc,
)


class ErrorType(Enum):
    TRAIN = auto()
    VALID = auto()
    EVAL = auto()


class Metrics:
    TESTING_ARRAY = np.asarray(
        [[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [6, 5, 4]], [[12, 3, 48], [54, 87, 7]]]
    )  # type: ignore

    def __init__(self, skip_check: bool = False, **metrics: _MetricType):
        self.metrics: Dict[str, MetricType] = STANDART_METRICS.copy()
        for name, value in metrics.items():
            if not isinstance(value, ca.Callable):  # type: ignore
                raise ValueError(f"'{name}' metric is not callable")
            metric = metric_adapter(value)
            if not self._check_metric(metric) and not skip_check:
                logger.warning(
                    "Metric [%s] does not meet the requirements of the axioms "
                    '[use skip_check=True" in Metrics] for skip this check',
                    name,
                )
            self.metrics[name] = metric
        self.results: Dict[ErrorType, Dict[str, List[MetricResult]]] = defaultdict(lambda: defaultdict(list))

    def calc(self, vec_true: np.ndarray, vec_pred: np.ndarray, error_type: ErrorType) -> None:
        for batch in vec_true:
            for metric_name, metric in self.metrics.items():
                result = metric(vec_true[batch], vec_pred[batch]) / len(vec_true[batch])
                self.results[error_type][metric_name].append(result)

    @classmethod
    def _check_metric(cls, metric: MetricType) -> bool:
        if metric(cls.TESTING_ARRAY[0], cls.TESTING_ARRAY[1]) != metric(cls.TESTING_ARRAY[1], cls.TESTING_ARRAY[0]):
            return False
        if metric(cls.TESTING_ARRAY[0], cls.TESTING_ARRAY[0]) != metric(cls.TESTING_ARRAY[0], cls.TESTING_ARRAY[0]):
            return False
        if metric(cls.TESTING_ARRAY[0], cls.TESTING_ARRAY[2]) > metric(
            cls.TESTING_ARRAY[0], cls.TESTING_ARRAY[1]
        ) + metric(cls.TESTING_ARRAY[1], cls.TESTING_ARRAY[2]):
            return False

        return True
