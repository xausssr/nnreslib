from __future__ import annotations

import collections.abc as ca
import functools
import logging
import math as m
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, Flag, auto, unique
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .categorial_metrics import (
    CATEGORIAL_METRICS,
    CalcCategorialMetrics,
    CategorialMetrics,
    CategorialMetricsAggregation,
)
from .categorial_metrics import MetricType as CategorialMetricType
from .regression_metrics import REGRESSION_METRICS
from .regression_metrics import MetricType as RegressionMetricType

_logger = logging.getLogger(__name__)


UserMetricType = Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], List[float]]
StandartMetricType = Union[RegressionMetricType, CategorialMetricType]
AllMetricType = Union[StandartMetricType, UserMetricType]

MetricResult = Union[float, np.ndarray, CategorialMetrics, List[CategorialMetrics]]

STANDART_METRICS: Dict[str, StandartMetricType] = {
    **REGRESSION_METRICS,
    **CATEGORIAL_METRICS,
}


@unique
class MetricFlags(Flag):
    NONE = 0
    MSE = auto()
    RMSE = auto()
    MAE = auto()
    ALL_REG = MSE | RMSE | MAE
    CCE = auto()
    CAT = auto()
    ALL_CAT = CCE | CAT


@unique
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
    def check(cls, **metrics: UserMetricType) -> None:
        for name, metric in metrics.items():
            if not isinstance(metric, ca.Callable):  # type: ignore
                raise ValueError(f"'{name}' metric is not callable")
            if not cls._check_metric(metric):
                _logger.warning(
                    "Metric [%s] does not meet the requirements of the axioms. "
                    "Metric still be using. "
                    '[use skip_check=True" in MetricsSettings] for skip this check.',
                    name,
                )

    @classmethod
    def _check_metric(cls, metric: UserMetricType) -> bool:
        if not m.isclose(
            metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_1,))[0], metric((cls.TEST_ARRAY_1,), (cls.TEST_ARRAY_0,))[0]
        ):
            return False
        if not m.isclose(
            metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_0,))[0], metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_0,))[0]
        ):
            return False
        if (
            metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_2,))[0]
            > metric((cls.TEST_ARRAY_0,), (cls.TEST_ARRAY_1,))[0] + metric((cls.TEST_ARRAY_1,), (cls.TEST_ARRAY_2,))[0]
        ):
            return False

        return True


@dataclass
class MetricsSettings:  # pylint:disable=too-many-instance-attributes
    standart_metrics: MetricFlags = MetricFlags.ALL_REG
    metrics_step: int = 1
    skip_check: bool = False
    user_metrics: Mapping[str, UserMetricType] = {}
    cat_thresholds: Optional[Union[float, List[float]]] = None
    reg_aggregate: bool = True
    cat_aggregate: Optional[CategorialMetricsAggregation] = None
    user_aggregate: Optional[Callable[[List[List[float]]], float]] = None

    def set_user_metrics(self, metrics: Dict[str, AllMetricType]) -> None:
        if not self.skip_check:
            MetricChecker.check(**self.user_metrics)
        for name, metric in self.user_metrics.items():
            metrics[name] = metric


_MetricInfo = NamedTuple("_MetricInfo", [("settings", MetricsSettings), ("metrics", Dict[str, AllMetricType])])


class BatchMetrics:
    __slots__ = ("_metrics_info", "_results", "_set_metrics_cb")

    def __init__(
        self,
        metrics_info: _MetricInfo,
        set_metrics_cb: Callable[[Iterable[Tuple[str, MetricResult]]], None],
    ) -> None:
        self._metrics_info = metrics_info
        self._results: Dict[str, List[Union[List[float], CalcCategorialMetrics]]] = defaultdict(list)
        self._set_metrics_cb = set_metrics_cb

    def __enter__(self) -> BatchMetrics:
        return self

    def __exit__(self, ex_type: Any, exp: Any, traceback: Any) -> bool:
        self._set_metrics_cb(self._reduce())
        return ex_type is None

    def _reduce(self) -> Generator[Tuple[str, MetricResult], None, None]:
        reg_agg = self._metrics_info.settings.reg_aggregate
        user_agg = self._metrics_info.settings.user_aggregate
        for metric_name, values in self._results.items():
            if isinstance(values[0], CalcCategorialMetrics):  # Categorical metric
                mean_value = sum(values[1:], values[0])
                yield metric_name, mean_value.get_metrics()  # type: ignore
            else:
                mean_value = np.mean(np.array(values), 0)  # type: ignore
                if metric_name not in STANDART_METRICS:  # User metric
                    yield metric_name, user_agg(mean_value) if user_agg else mean_value  # type: ignore
                else:  # Regression metric
                    yield metric_name, np.mean(mean_value) if reg_agg else mean_value  # type: ignore

    # TODO Separate metrics to each output
    def calc_batch(self, vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> None:
        for metric_name, metric in self._metrics_info.metrics.items():
            self._results[metric_name].append(metric(vec_true, vec_pred))


class EmptyBatchMetrics(BatchMetrics):
    def __init__(self) -> None:  # pylint:disable=super-init-not-called
        ...

    def __exit__(self, ex_type: Any, exp: Any, traceback: Any) -> bool:
        return ex_type is None

    def calc_batch(self, vec_true: Sequence[np.ndarray], vec_pred: Sequence[np.ndarray]) -> None:
        ...


class Metrics:

    __slots__ = ("_metrics", "results")

    def __init__(self) -> None:
        self._metrics: Dict[OpMode, _MetricInfo] = {}
        # TODO: think about metric result type
        self.results: Dict[OpMode, Dict[str, List[MetricResult]]] = defaultdict(lambda: defaultdict(list))

    def set_settings(self, op_mode: OpMode, settings: Optional[MetricsSettings]) -> None:
        metric_info = self._metrics.get(op_mode)
        if metric_info is None or settings is not None:
            op_settings = settings if settings is not None else MetricsSettings()
            metrics = {
                name: value
                for name, value in STANDART_METRICS.items()
                if MetricFlags[name] & op_settings.standart_metrics
            }
            if MetricFlags.CAT & op_settings.standart_metrics:
                # pylint:disable=no-member # Due to pylint bug https://github.com/PyCQA/pylint/issues/533
                metrics[MetricFlags.CAT.name] = functools.partial(  # type: ignore
                    metrics[MetricFlags.CAT.name],
                    thresholds=op_settings.cat_thresholds,
                    aggregate=op_settings.cat_aggregate,
                )
                # pylint:enable=no-member
            op_settings.set_user_metrics(metrics)
            self._metrics[op_mode] = _MetricInfo(op_settings, metrics)

    def batch_metrics(self, op_mode: OpMode, epoch: int) -> BatchMetrics:
        metric_info = self._metrics[op_mode]
        if epoch % metric_info.settings.metrics_step != 0:
            return EmptyBatchMetrics()
        return BatchMetrics(metric_info, functools.partial(self.set_batch_metrics, op_mode=op_mode))

    def set_batch_metrics(
        self,
        metrics_results: Iterable[Tuple[str, MetricResult]],
        op_mode: OpMode,
    ) -> None:
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
