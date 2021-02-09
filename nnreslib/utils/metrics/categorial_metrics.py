from __future__ import annotations

import functools
import math as m
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable, Dict, List, Optional, Sequence, Type, Union, overload

import numpy as np


def _cce(vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> float:
    # TODO: add some checks for vec_true
    result = 0.0
    for out_true, out_pred in zip(vec_true, vec_pred):
        result += np.mean(-(out_true * np.log(out_pred)))
    return result / len(vec_true)


@dataclass
class CategorialMetrics:
    # pylint: disable=invalid-name
    TP: int = 0
    FP: int = 0
    FN: int = 0
    TN: int = 0

    Prevalence: float = 0.0
    Accuracy: float = 0.0  # ACC

    Precision: float = 0.0  # Positive predictive value (PPV)
    FDR: float = 0.0  # False discovery rate
    FOR: float = 0.0  # False omission rate
    NPV: float = 0.0  # Negative predictive value

    Recall: float = 0.0
    FPR: float = 0.0  # False positive rate, Fall-out
    FNR: float = 0.0  # False negative rate, Miss rate
    Selectivity: float = 0.0  # True negative rate (TNR), Specificity (SPC)

    LR_positive: float = 0.0  # Positive likelihood ratio
    LR_negative: float = 0.0  # Negative likelihood ratio
    DOR: float = 0.0  # Diagnostic odds ratio
    F1_score: float = 0.0

    BM: float = 0.0  # TODO: describe
    PT: float = 0.0  # TODO: describe
    TS: float = 0.0  # TODO: describe
    BA: float = 0.0  # TODO: describe
    MCC: float = 0.0  # TODO: describe
    FM: float = 0.0  # TODO: describe
    MK: float = 0.0  # TODO: describe

    ROC_AUC_cycle: float = 0.0  # TODO: describe
    ROC_AUC_analytic: float = 0.0  # TODO: describe
    ROC: float = 0.0  # TODO: describe
    PRC: float = 0.0  # TODO: describe
    PR_AUC: float = 0.0  # TODO: describe

    # pylint: enable=invalid-name

    def __add__(self, value: object) -> CategorialMetrics:
        if not isinstance(value, CategorialMetrics):
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(value).__name__}'")
        return CategorialMetrics(
            self.TP + value.TP,
            self.FP + value.FP,
            self.FN + value.FN,
            self.TN + value.TN,
        )

    def __iadd__(self, value: object) -> CategorialMetrics:
        if not isinstance(value, CategorialMetrics):
            raise TypeError(f"unsupported operand type(s) for +=: '{type(self).__name__}' and '{type(value).__name__}'")
        self.TP += value.TP
        self.FP += value.FP
        self.FN += value.FN
        self.TN += value.TN
        return self

    def calc_metrics(self) -> None:
        def save_div(value_a: Union[float, int], value_b: Union[float, int]) -> float:
            try:
                return value_a / value_b
            except ZeroDivisionError:
                return 1

        condition_positive = self.TP + self.FN
        condition_negative = self.FP + self.TN
        predicted_condition_positive = self.TP + self.FP
        predicted_condition_negative = self.FN + self.TN
        total = condition_positive + condition_negative

        self.Prevalence = save_div(self.TP + self.FP, total)
        self.Accuracy = save_div(self.TP + self.TN, total)
        self.Precision = save_div(self.TP, predicted_condition_positive)
        self.FDR = save_div(self.FP, predicted_condition_positive)
        self.FOR = save_div(self.FN, predicted_condition_negative)
        self.NPV = save_div(self.TN, predicted_condition_negative)

        self.Recall = save_div(self.TP, condition_positive)
        self.FPR = save_div(self.FP, condition_negative)
        self.FNR = save_div(self.FN, condition_positive)
        self.Selectivity = save_div(self.TN, condition_negative)

        self.LR_positive = save_div(self.Recall, self.FPR)
        self.LR_negative = save_div(self.FNR, self.Selectivity)
        self.DOR = save_div(self.LR_positive, self.LR_negative)
        self.F1_score = save_div(2 * self.Precision * self.Recall, self.Precision + self.Recall)

        self.BM = self.Recall + self.Selectivity - 1
        self.PT = save_div(m.sqrt(self.Recall * (1 - self.Selectivity)) + self.Selectivity - 1, self.BM)
        self.TS = save_div(self.TP, condition_positive + self.FP)
        self.BA = (self.Recall + self.Selectivity) / 2
        self.MCC = save_div(
            self.TP * self.TN - self.FP * self.FN,
            m.sqrt(
                condition_positive * condition_negative * predicted_condition_positive * predicted_condition_negative
            ),
        )
        self.FM = m.sqrt(self.Precision * self.Recall)
        self.MK = self.Precision + self.NPV - 1

    @classmethod
    def get(cls, vec_true: np.ndarray, vec_pred: np.ndarray, threshold: float = 0.5) -> CategorialMetrics:
        if not ((vec_true == 0) | (vec_true == 1)).all():
            raise ValueError("Calculation categorial metrics for regression data!")

        vec_pred_norm = (vec_pred >= threshold).astype(int)
        diff = vec_true - vec_pred_norm
        cond_pos = np.asarray(vec_true == 1)  # type: ignore
        cond_neg = np.asarray(vec_true == 0)  # type: ignore
        pred_cond_pos = np.asarray(vec_pred_norm == 1)  # type: ignore
        pred_cond_neg = np.asarray(vec_pred_norm == 0)  # type: ignore
        return CategorialMetrics(
            TP=np.count_nonzero(cond_pos * pred_cond_pos, 0),  # type: ignore
            FP=np.count_nonzero(np.asarray(diff == -1), 0),  # type: ignore
            FN=np.count_nonzero(np.asarray(diff == 1), 0),  # type: ignore
            TN=np.count_nonzero(cond_neg * pred_cond_neg, 0),  # type: ignore
        )


@unique
class CategorialMetricsAggregation(Enum):
    MEAN = functools.partial(np.mean)
    MAX = functools.partial(max)
    MIN = functools.partial(min)


class CalcCategorialMetrics:
    __slots__ = ("_metrics", "_aggregate")

    @overload
    def __init__(
        self,
        vec_true: Sequence[np.ndarray],
        vec_pred: Sequence[np.ndarray],
        thresholds: Optional[Union[float, List[float]]] = None,
        aggregate: Optional[CategorialMetricsAggregation] = None,
    ) -> None:
        "By default threshold will be 0.5 for every out"
        ...

    @overload
    def __init__(
        self,
        *,
        aggregate: Optional[CategorialMetricsAggregation] = None,
        metrics: List[CategorialMetrics],
    ) -> None:
        ...

    def __init__(
        self,
        vec_true: Optional[Sequence[np.ndarray]] = None,
        vec_pred: Optional[Sequence[np.ndarray]] = None,
        thresholds: Optional[Union[float, List[float]]] = None,
        aggregate: Optional[CategorialMetricsAggregation] = None,
        metrics: Optional[List[CategorialMetrics]] = None,
    ) -> None:
        self._aggregate = aggregate
        if metrics is not None:
            self._metrics = metrics
        else:
            if vec_true is None or vec_pred is None:
                raise ValueError("Empty 'vec_true' or 'vec_pred'")
            _thresholds = CalcCategorialMetrics._get_thresholds(len(vec_true), thresholds)
            # FIXME: think about batch conf_matrix aggregation (sum conf_matrix value)
            self._metrics = [
                CategorialMetrics.get(y_true, y_pred, threshold)
                for y_true, y_pred, threshold in zip(vec_true, vec_pred, _thresholds)
            ]
        # if aggregate:
        #     self._metrics = CategorialMetrics()
        #     for y_true, y_pred in zip(vec_true, vec_pred):
        #         self._metrics += CategorialMetrics.get(y_true, y_pred, threshold)
        # else:
        #     self._metrics = []
        #     for y_true, y_pred in zip(vec_true, vec_pred):
        #         self._metrics.append(CategorialMetrics.get(y_true, y_pred, threshold))

    # TODO: move it to utils. Also used in ForwardGraph (_get_thresholds)
    @staticmethod
    def _get_thresholds(out_count: int, thresholds: Optional[Union[float, List[float]]]) -> List[float]:
        if thresholds is None:
            _thresholds = [0.5 for _ in range(out_count)]
        else:
            if isinstance(thresholds, float):
                _thresholds = [thresholds]
            else:
                _thresholds = thresholds
        return _thresholds

    def __add__(self, value: object) -> CalcCategorialMetrics:
        if not isinstance(value, CalcCategorialMetrics):
            raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(value).__name__}'")
        return CalcCategorialMetrics(
            aggregate=self._aggregate,
            metrics=[x + y for x, y in zip(self._metrics, value._metrics)],
        )

    def __iadd__(self, value: object) -> CalcCategorialMetrics:
        if not isinstance(value, CalcCategorialMetrics):
            raise TypeError(f"unsupported operand type(s) for +=: '{type(self).__name__}' and '{type(value).__name__}'")
        for metric, v_metric in zip(self._metrics, value._metrics):
            metric += v_metric
        return self

    def __call__(self) -> CategorialMetrics:
        ...


_MetricType1 = Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], float]
# _MetricType2 = Callable[..., CategorialMetrics]
_MetricType2 = Type[CalcCategorialMetrics]
MetricType = Union[_MetricType1, _MetricType2]

CATEGORIAL_METRICS: Dict[str, MetricType] = dict(
    CCE=_cce,
    CAT=CalcCategorialMetrics,
)
