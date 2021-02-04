from __future__ import annotations

import collections.abc as ca
import functools
import logging
import math as m
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, Iterable, List, Sequence, Tuple

import numpy as np

_logger = logging.getLogger(__name__)


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


MetricType = Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], float]

# def metric_adapter(func: _MetricType) -> MetricType:
#     def adapter(vec_true: np.ndarray, vec_pred: np.ndarray) -> MetricResult:
#         return MetricResult(func(vec_true, vec_pred))

#     return adapter


# @metric_adapter
def __mse(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray], sqrt_func: Callable[[float], float]) -> float:
    result = 0.0
    for out_true, out_pred in zip(vec_true, vec_pred):
        result += sqrt_func(np.mean((out_true - out_pred) ** 2))
    return result / len(vec_true)


def _mse(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> float:
    return __mse(vec_true, vec_pred, lambda x: x)


def _rmse(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> float:
    return __mse(vec_true, vec_pred, np.sqrt)


# @metric_adapter
def _mae(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> float:
    result = 0.0
    for out_true, out_pred in zip(vec_true, vec_pred):
        result += np.mean(np.abs(out_true - out_pred))
    return result / len(vec_true)


# @metric_adapter
def _cce(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> float:
    # TODO: add some checks for vec_true
    result = 0.0
    for out_true, out_pred in zip(vec_true, vec_pred):
        result += np.mean(-(out_true * np.log(out_pred)))
    return result / len(vec_true)


# @metric_adapter
def _roc(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> np.ndarray:
    raise NotImplementedError()


# @metric_adapter
def _auc(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> np.ndarray:
    raise NotImplementedError()


STANDART_METRICS: Dict[str, MetricType] = dict(
    MSE=_mse,
    RMSE=_rmse,
    MAE=_mae,
    CCE=_cce,
    # ROC=_roc,
    # AUC=_auc,
)


class OpMode(Enum):
    TRAIN = auto()
    VALID = auto()
    EVAL = auto()


class MetricChecker:
    __slots__ = ()

    TEST_ARRAY_0 = np.array([[1, 2, 3], [3, 2, 1]])
    TEST_ARRAY_1 = np.array([[4, 5, 6], [6, 5, 4]])
    TEST_ARRAY_2 = np.array([[12, 3, 48], [54, 87, 7]])

    @classmethod
    def check(cls, **metrics: MetricType) -> None:
        for name, metric in metrics.items():
            if not isinstance(metric, ca.Callable):  # type: ignore
                raise ValueError(f"'{name}' metric is not callable")
            if not cls._check_metric(metric):
                _logger.warning(
                    "Metric [%s] does not meet the requirements of the axioms. "
                    "Metric still be using. "
                    '[use skip_check=True" in Metrics] for skip this check.',
                    name,
                )

    @classmethod
    def _check_metric(cls, metric: MetricType) -> bool:
        if metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_1,)) != metric((cls.TEST_ARRAY_1,), (cls.TEST_ARRAY_0,)):
            return False
        if metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_0,)) != metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_0,)):
            return False
        if metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_2,)) > metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_1,)) + metric(
            (cls.TEST_ARRAY_1,), (cls.TEST_ARRAY_2,)
        ):
            return False

        return True


class BatchMetrics:
    def __init__(
        self, metrics: Dict[str, MetricType], set_metrics_cb: Callable[[Iterable[Tuple[str, float]]], None]
    ) -> None:
        self._metrics = metrics
        self._result: Dict[str, List[float]] = defaultdict(list)
        self._set_metrics_cb = set_metrics_cb

    def __enter__(self) -> BatchMetrics:
        return self

    def __exit__(self, ex_type: Any, exp: Any, traceback: Any) -> bool:
        self._set_metrics_cb(self._reduce())
        return ex_type is None

    def _reduce(self) -> Generator[Tuple[str, float], None, None]:
        return ((metric_name, np.mean(np.array(value))) for metric_name, value in self._result.items())

    # TODO Separate metrics to each output
    def calc_batch(self, vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> None:
        for metric_name, metric in self._metrics.items():
            self._result[metric_name].append(metric(vec_true, vec_pred))


class EmptyBatchMetrics(BatchMetrics):
    def __init__(self) -> None:  # pylint:disable=super-init-not-called
        ...

    def __exit__(self, ex_type: Any, exp: Any, traceback: Any) -> bool:
        return ex_type is None

    def calc_batch(self, vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> None:
        ...


class Metrics:

    __slots__ = ("_metrics_step", "_metrics", "results")

    def __init__(self, metrics_step: int = 1, skip_check: bool = False, **metrics: MetricType):
        self._metrics_step = metrics_step
        self._metrics: Dict[str, MetricType] = STANDART_METRICS.copy()
        if not skip_check:
            MetricChecker.check(**metrics)
        for name, metric in metrics.items():
            self._metrics[name] = metric
        # TODO: think about metric result type
        self.results: Dict[OpMode, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def clear(self, *op_mode: OpMode) -> None:
        for mode in op_mode:
            self.results[mode].clear()

    def batch_metrics(self, op_mode: OpMode, epoch: int) -> BatchMetrics:
        if epoch % self._metrics_step == 0:
            return BatchMetrics(self._metrics, functools.partial(self.set_batch_metrics, op_mode=op_mode))
        return EmptyBatchMetrics()

    def set_batch_metrics(self, metrics_results: Iterable[Tuple[str, float]], op_mode: OpMode) -> None:
        for metric_name, value in metrics_results:
            self.results[op_mode][metric_name].append(value)

    def _conf_matrix(self, data: Tuple[np.ndarray, np.ndarray], treshold: float = 0.5) -> dict:
        classification_metrics = {"TP": 0.0, "FP": 0.0, "FN": 0.0, "TN": 0.0}
        diff = data[0] - (data[1] > treshold)
        classification_metrics["TP"] += len(np.where(data[0] == 1)[0]) - np.sum(diff[np.where(data[0] == 1)[0]])
        classification_metrics["FP"] += np.sum(diff[np.where(data[0] == 1)[0]])
        classification_metrics["FN"] += len(np.where(diff < 0)[0])
        classification_metrics["TN"] += len(np.where(diff[np.where(data[0] == 0)[0]] == 0)[0])
        return classification_metrics

    def confusion_data(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        score_beta: Tuple[float] = (1.0,),
        treshold: float = 0.5,
        curves: bool = True,
    ) -> dict:

        classification_metrics = self._conf_matrix(data, treshold)

        condition_negative = classification_metrics["FP"] + classification_metrics["TN"]
        condition_positive = classification_metrics["TP"] + classification_metrics["FN"]
        predicted_condition_positive = classification_metrics["TP"] + classification_metrics["FP"]
        predicted_condition_negative = classification_metrics["TN"] + classification_metrics["FN"]
        total_population = len(data[0])

        classification_metrics["Recall"] = classification_metrics["TP"] / condition_positive
        classification_metrics["FPR"] = classification_metrics["FP"] / condition_negative
        classification_metrics["FNR"] = classification_metrics["FN"] / condition_positive
        classification_metrics["TNR"] = classification_metrics["TN"] / condition_negative

        classification_metrics["Prevalence"] = condition_positive / total_population
        classification_metrics["Accuracy"] = (
            classification_metrics["TP"] + classification_metrics["TN"]
        ) / total_population
        classification_metrics["Precision"] = classification_metrics["TP"] / predicted_condition_positive
        classification_metrics["FDR"] = classification_metrics["FP"] / predicted_condition_positive
        classification_metrics["FOR"] = classification_metrics["FN"] / predicted_condition_negative
        classification_metrics["NPV"] = classification_metrics["TN"] / predicted_condition_negative

        try:
            classification_metrics["LR+"] = classification_metrics["Recall"] / classification_metrics["FPR"]
        except ZeroDivisionError:
            classification_metrics["LR+"] = 1

        try:
            classification_metrics["LR-"] = classification_metrics["FNR"] / classification_metrics["TNR"]
        except ZeroDivisionError:
            classification_metrics["LR-"] = 1
        try:
            classification_metrics["DOR"] = classification_metrics["LR+"] / classification_metrics["LR-"]
        except ZeroDivisionError:
            classification_metrics["DOR"] = 1
        try:
            classification_metrics["PT"] = (
                m.sqrt(classification_metrics["Recall"] * (1 - classification_metrics["TNR"]))
                + classification_metrics["TNR"]
                - 1
            ) / (classification_metrics["Recall"] + classification_metrics["TNR"] - 1)
        except ZeroDivisionError:
            classification_metrics["PT"] = 1
        classification_metrics["TS"] = classification_metrics["TP"] / (
            classification_metrics["TP"] + classification_metrics["FN"] + classification_metrics["FP"]
        )

        classification_metrics["BA"] = (classification_metrics["Recall"] + classification_metrics["TNR"]) / 2

        classification_metrics["MCC"] = (
            classification_metrics["TP"] * classification_metrics["TN"]
            - classification_metrics["FP"] * classification_metrics["FN"]
        ) / (
            m.sqrt(
                (classification_metrics["TP"] + classification_metrics["FP"])
                * (classification_metrics["TP"] + classification_metrics["FN"])
                * (classification_metrics["TN"] + classification_metrics["FP"])
                * (classification_metrics["TN"] + classification_metrics["FN"])
            )
        )
        classification_metrics["FM"] = m.sqrt(classification_metrics["Precision"] * classification_metrics["Recall"])

        classification_metrics["BM"] = classification_metrics["Recall"] + classification_metrics["TNR"] - 1
        classification_metrics["MK"] = classification_metrics["Precision"] + classification_metrics["NPV"] - 1

        for beta in score_beta:
            classification_metrics["F" + str(beta)] = (1.0 * beta ** 2 * classification_metrics["TP"]) / (
                (1 + beta ** 2) * classification_metrics["TP"]
                + beta ** 2 * classification_metrics["FN"]
                + classification_metrics["FP"]
            )

        if not curves:
            return classification_metrics

        ones = np.sum(data[0])
        sorted_labels = np.hstack([data[1], data[0]])
        sorted_labels = sorted_labels[np.argsort(sorted_labels[:, 0])][::-1]

        roc = [0]
        prc = np.zeros(shape=(2, 100))
        temp_roc_value = 0
        auc = 0

        for label in sorted_labels:
            if label[0] == 1:
                temp_roc_value += 1.0 / ones
            else:
                roc.append(temp_roc_value)
                auc += temp_roc_value

        idx = 0
        for temp_treshold in np.linspace(0, 1, num=prc.shape[1]):
            temp_matrix = self._conf_matrix(data, temp_treshold)

            try:
                prc[0, idx] = temp_matrix["TP"] / (temp_matrix["TP"] + temp_matrix["FP"])
            except ZeroDivisionError:
                prc[0, idx] = 1.0
            try:
                prc[1, idx] = temp_matrix["TP"] / (temp_matrix["TP"] + temp_matrix["FN"])
            except ZeroDivisionError:
                prc[1, idx] = 1.0

            idx += 1

        classification_metrics["ROC_AUC_cycle"] = auc / (len(sorted_labels) - ones)
        classification_metrics["ROC_AUC_analytic"] = (
            1 + classification_metrics["Recall"] - classification_metrics["FPR"]
        ) / 2
        classification_metrics["ROC"] = roc
        classification_metrics["PRC"] = prc
        classification_metrics["PR_AUC"] = np.sum(prc[0, :]) / prc.shape[1]

        return classification_metrics

    def multimodal_confusion_data(
        self, data: Tuple[np.ndarray, np.ndarray], score_beta: Tuple[float] = (1.0,), treshold: float = 0.5
    ) -> dict:
        classification_metrics: Dict = {}
        for class_idx in range(len(data[0][0])):
            classification_metrics[f"Class {class_idx}"] = {}
            temp_result = self.confusion_data(
                (data[0][:, class_idx].reshape((-1, 1)), data[1][:, class_idx].reshape((-1, 1))),
                score_beta=score_beta,
                treshold=treshold,
            )
            for key in temp_result:
                classification_metrics[f"Class {class_idx}"][key] = temp_result[key]

        classification_metrics["average"] = {}

        # classification_metrics["class 1", "class 2", ...] -> mean(classification_metrics)

        temp_result = self.confusion_data(
            (data[0].ravel().reshape((-1, 1)), data[1].ravel().reshape((-1, 1))),
            score_beta=score_beta,
            treshold=treshold,
        )
        for key in temp_result:
            classification_metrics["average"][key] = temp_result[key]

        return classification_metrics
