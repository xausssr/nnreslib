from __future__ import annotations

import collections.abc as ca
import logging
import math as m
from collections import defaultdict
from enum import Enum, auto
from typing import Callable, Dict, List, Tuple

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
    _logger.warning("Metric MAE is categorical! You need use 2D one-hot encoded arrays for output result")
    return np.zeros(1)


@metric_adapter
def _cce(vec_true: np.ndarray, vec_pred: np.ndarray) -> np.ndarray:
    if len(vec_true.shape) == 2 and len(vec_pred.shape) == 2:
        return np.array([np.sum(-(vec_true * np.log(vec_pred)))])
    _logger.warning(
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
                _logger.warning(
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

    def _conf_matrix(self, data: Tuple[np.ndarray, np.ndarray], treshold: float = 0.5) -> dict:
        classification_metrics = {"TP": 0.0, "FP": 0.0, "FN": 0.0, "TN": 0.0}
        for label in range(len(data[0])):
            if data[0][label] == int(data[1][label] > treshold):
                if data[0][label] == 0:
                    classification_metrics["TN"] += 1.0
                else:
                    classification_metrics["TP"] += 1.0
            else:
                if data[0][label] == 0:
                    classification_metrics["FN"] += 1.0
                else:
                    classification_metrics["FP"] += 1.0
        return classification_metrics

    def confusion_data(
        self, data: Tuple[np.ndarray, np.ndarray], score_beta: Tuple[float] = (1.0,), treshold: float = 0.5
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

        ones = np.sum(data[0])
        sorted_labels = np.hstack([data[1], data[0]])
        sorted_labels = sorted_labels[np.argsort(sorted_labels[:, 0])][::-1]

        roc = [0]
        prc = np.zeros(shape=(2, 100))
        temp_roc_value = 0
        auc = 0

        for pos in range(len(sorted_labels)):
            if sorted_labels[pos][1] == 1:
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
            for key in temp_result.keys():
                classification_metrics[f"Class {class_idx}"][key] = temp_result[key]

        classification_metrics["average"] = {}
        temp_result = self.confusion_data(
            (data[0].ravel().reshape((-1, 1)), data[1].ravel().reshape((-1, 1))),
            score_beta=score_beta,
            treshold=treshold,
        )
        for key in temp_result.keys():
            classification_metrics["average"][key] = temp_result[key]

        return classification_metrics


# classification_metrics(data=(y, y_hat), ...) -> y.shape[1] == 1: confusion_data; multimodal_confusion_data
# len(y.shape) != 2: error; y.shape != y_hat.shape: error; treshold in (0,1)
