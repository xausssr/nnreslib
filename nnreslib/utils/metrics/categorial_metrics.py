from __future__ import annotations

import functools
import math as m
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

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
        if isinstance(value, CategorialMetrics):
            return CategorialMetrics(
                self.TP + value.TP,
                self.FP + value.FP,
                self.FN + value.FN,
                self.TN + value.TN,
            )
        raise TypeError(f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(value).__name__}'")

    def iadd(self, value: object) -> None:
        if isinstance(value, CategorialMetrics):
            self.TP += value.TP
            self.FP += value.FP
            self.FN += value.FN
            self.TN += value.TN
        raise TypeError(f"unsupported operand type(s) for +=: '{type(self).__name__}' and '{type(value).__name__}'")

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
        vec_pred_norm = (vec_pred >= threshold).astype(int)
        diff = vec_true - vec_pred_norm
        cond_pos = np.asarray(vec_true == 1)  # type: ignore
        cond_neg = np.asarray(vec_true == 0)  # type: ignore
        pred_cond_pos = np.asarray(vec_pred == 1)  # type: ignore
        pred_cond_neg = np.asarray(vec_pred == 0)  # type: ignore
        return CategorialMetrics(
            np.count_nonzero(cond_pos * pred_cond_pos),  # type: ignore
            np.count_nonzero(np.asarray(diff == -1)),  # type: ignore
            np.count_nonzero(np.asarray(diff == 1)),  # type: ignore
            np.count_nonzero(cond_neg * pred_cond_neg),  # type: ignore
        )


@unique
class CategorialMetricsAggregation(Enum):
    MEAN = functools.partial(np.mean)
    MAX = functools.partial(max)
    MIN = functools.partial(min)


class CalcCategorialMetrics:
    __slots__ = ("_metrics",)

    def __init__(
        self,
        vec_true: Sequence[np.ndarray],
        vec_pred: Sequence[np.ndarray],
        threshold: float = 0.5,
        aggregate: Optional[CategorialMetricsAggregation] = None,
    ) -> None:
        self._metrics: Union[CategorialMetrics, List[CategorialMetrics]]
        self._metrics = []
        for y_true, y_pred in zip(vec_true, vec_pred):
            self._metrics.append(CategorialMetrics.get(y_true, y_pred, threshold))
        if aggregate:
            self._metrics = CategorialMetrics()
            for y_true, y_pred in zip(vec_true, vec_pred):
                self._metrics += CategorialMetrics.get(y_true, y_pred, threshold)
        else:
            self._metrics = []
            for y_true, y_pred in zip(vec_true, vec_pred):
                self._metrics.append(CategorialMetrics.get(y_true, y_pred, threshold))

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
