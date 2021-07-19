import logging
import re

import numpy as np
import pytest

from nnreslib.utils.metrics import MetricFlags, Metrics, MetricsSettings, OpMode
from nnreslib.utils.metrics.categorial_metrics import (
    CalcCategorialMetrics,
    CategorialMetricsAggregation,
)
from nnreslib.utils.metrics.metrics import STANDART_METRICS, BatchMetrics, EmptyBatchMetrics
from nnreslib.utils.metrics.regression_metrics import REGRESSION_METRICS, rmse

np.random.seed(42)


def test_metric_flags():
    for metric_name in STANDART_METRICS:
        MetricFlags[metric_name]  # pylint:disable=pointless-statement


def test_metrics_settings(caplog):
    # No user metrics
    metrics = {}
    settings = MetricsSettings()
    settings.set_user_metrics(metrics)
    assert not metrics

    # Good user metrics
    settings = MetricsSettings(user_metrics=dict(MY_RMSE=rmse))
    settings.set_user_metrics(metrics)
    assert "MY_RMSE" in metrics

    # Non callable user metric
    settings = MetricsSettings(user_metrics=dict(NON_CALL=10))
    with pytest.raises(ValueError, match=r"NON_CALL.*is not callable"):
        settings.set_user_metrics(metrics)

    # Bad user metric
    def bad_metric(vec_true, vec_pred):
        return [np.sum(np.less(out_true, out_pred)) for out_true, out_pred in zip(vec_true, vec_pred)]

    settings = MetricsSettings(user_metrics=dict(BAD_METRIC=bad_metric))
    settings.set_user_metrics(metrics)
    assert "BAD_METRIC" in metrics
    _, level, message = caplog.record_tuples[0]
    assert level == logging.WARNING
    assert re.match(r"^Metric \[BAD_METRIC\] does not meet", message) is not None

    # TODO: check other metrics axiom


def test_set_settings():
    # TODO: check results clearing
    metrics = Metrics()

    # pylint:disable=protected-access

    # Empty metrics, empty settings
    settings = None
    metrics.set_settings(OpMode.TRAIN, settings)
    assert metrics._metrics[OpMode.TRAIN].settings == MetricsSettings()
    for metric_name in REGRESSION_METRICS:
        assert metric_name in metrics._metrics[OpMode.TRAIN].metrics

    # Not empty metrics, empty settings
    old_metrics = metrics._metrics
    metrics.set_settings(OpMode.TRAIN, settings)
    assert metrics._metrics is old_metrics

    # Non empty settings (empty metrics or not: result is the same)
    settings = MetricsSettings(
        standart_metrics=MetricFlags.ALL_CAT, cat_thresholds=0.5, cat_aggregate=CategorialMetricsAggregation.MEAN
    )
    metrics.set_settings(OpMode.TRAIN, settings)
    # pylint:disable=no-member
    # XXX: Due to pylint bug https://github.com/PyCQA/pylint/issues/533
    assert MetricFlags.CCE.name in metrics._metrics[OpMode.TRAIN].metrics
    cat_metrics = metrics._metrics[OpMode.TRAIN].metrics[MetricFlags.CAT.name]
    # pylint:enable=no-member
    assert cat_metrics.func is CalcCategorialMetrics
    assert cat_metrics.keywords["thresholds"] == 0.5
    assert cat_metrics.keywords["aggregate"] == CategorialMetricsAggregation.MEAN

    # pylint:disable=protected-access


def test_creation_batch_metrics():
    metrics = Metrics()
    settings_train = MetricsSettings(metrics_step=10)
    settings_valid = MetricsSettings(metrics_step=20)

    metrics.set_settings(OpMode.TRAIN, settings_train)
    metrics.set_settings(OpMode.VALID, settings_valid)

    assert isinstance(metrics.batch_metrics(OpMode.TRAIN, 0), BatchMetrics)
    assert isinstance(metrics.batch_metrics(OpMode.TRAIN, 5), EmptyBatchMetrics)
    assert isinstance(metrics.batch_metrics(OpMode.TRAIN, 10), BatchMetrics)

    assert isinstance(metrics.batch_metrics(OpMode.VALID, 0), BatchMetrics)
    assert isinstance(metrics.batch_metrics(OpMode.VALID, 5), EmptyBatchMetrics)
    assert isinstance(metrics.batch_metrics(OpMode.VALID, 20), BatchMetrics)


def test_empty_batch_metrics():
    # pylint:disable=protected-access,pointless-statement
    with EmptyBatchMetrics() as ebm:
        ebm.calc_batch((np.array([1, 2, 3]),), (np.array([5, 6, 7]),))
        with pytest.raises(AttributeError, match="_metrics_info"):
            ebm._metrics_info
        with pytest.raises(AttributeError, match="_results"):
            ebm._results
        with pytest.raises(AttributeError, match="_set_metrics_cb"):
            ebm._set_metrics_cb
    # pylint:enable=protected-access,pointless-statement


def test_calc_batch_metrics():
    metrics = Metrics()

    settings = MetricsSettings(
        metrics_step=2,
        standart_metrics=MetricFlags.MSE | MetricFlags.MAE,
        user_metrics=dict(CUSTOM_RMSE=rmse),
        reg_aggregate=False,
    )  # All metrics without aggregation
    metrics.set_settings(OpMode.TRAIN, settings)

    settings = MetricsSettings(
        metrics_step=4,
        standart_metrics=MetricFlags.MSE | MetricFlags.MAE,
        user_metrics=dict(CUSTOM_RMSE=rmse),
        cat_aggregate=CategorialMetricsAggregation.MAX,
        user_aggregate=np.sum,
    )  # All metrics with aggregation
    metrics.set_settings(OpMode.VALID, settings)

    for epoch in range(10):
        with metrics.batch_metrics(OpMode.TRAIN, epoch) as batch_metrics:
            batch_metrics.calc_batch((), ())
        with metrics.batch_metrics(OpMode.VALID, epoch) as batch_metrics:
            batch_metrics.calc_batch((), ())

    # for CAT metrics
    # np.random.randint(1, 5, 10) -> array([1, 3, 4, 1, 2, 1, 3, 4, 1, 2])

    # for regression metrics
    # np.random.rand(2,5) -> array([[0.95928577, 0.91874734, 0.45869854, 0.64542553, 0.41451908],
    #                                     [0.49299998, 0.01529939, 0.40237871, 0.1525967 , 0.86286436]])

    # assert len(metrics.results[OpMode.TRAIN]["MSE"]) == len(metrics.results[OpMode.TRAIN]["CAT"])
    assert len(metrics.results[OpMode.TRAIN]["MSE"]) == 5
    assert len(metrics.results[OpMode.VALID]["MSE"]) == 3
    # assert len(metrics.results[OpMode.TRAIN]["MSE"]) == len(metrics.results[OpMode.TRAIN]["CAT"])
    # assert len(metrics.results[OpMode.TRAIN]["MSE"]) == 5

    # metrics.results[OpMode.TRAIN]["MSE"]
    metrics.set_settings(OpMode.VALID, settings)
    assert len(metrics.results[OpMode.VALID]) == 0
