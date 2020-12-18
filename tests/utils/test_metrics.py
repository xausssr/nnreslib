import logging
import re

import numpy as np
import pytest

from nnreslib.utils.metrics import MetricResult, Metrics, _mse


def test_metric_result_add():
    left_generation = np.random.rand(1, 150)
    right_generation = np.random.rand(1, 150)
    left = MetricResult(left_generation)
    right = MetricResult(right_generation)
    assert np.array_equal((left + right).out_values, left_generation + right_generation)
    assert np.array_equal((right + left).out_values, right_generation + left_generation)


def test_metrics_result_eq():
    data = np.random.rand(1, 150)
    left = MetricResult(data)
    right = MetricResult(data)
    assert left == right
    assert right == left
    assert left != 861
    assert left != 86.1
    assert left != "__861__"


def test_metric_result_lt():
    left_generation = np.random.rand(1, 150)
    right_generation = np.random.rand(1, 150) + 861
    left = MetricResult(left_generation)
    right = MetricResult(right_generation)
    assert left < right
    assert right > left
    with pytest.raises(TypeError, match=r"not supported.*'MetricResult'.*'int'"):
        left < 861  # pylint:disable=pointless-statement
    with pytest.raises(TypeError, match=r"not supported.*'MetricResult'.*'float'"):
        left < 86.1  # pylint:disable=pointless-statement
    with pytest.raises(TypeError, match=r"not supported.*'MetricResult'.*'str'"):
        left < "__861__"  # pylint:disable=pointless-statement


def test_metric_result_truediv():
    data = np.random.rand(1, 150)
    metric_result = MetricResult(data)
    assert np.array_equal((metric_result / 861).out_values, data / 861)
    assert np.array_equal((metric_result / 86.1).out_values, data / 86.1)
    with pytest.raises(TypeError, match=r"unsupported.*'MetricResult'.*'str'"):
        metric_result / "__861__"  # pylint:disable=pointless-statement


def test_metrics_init(caplog):
    Metrics(MY_MSE=_mse)
    assert not caplog.records

    def bad_metric(vec_true, vec_pred):
        return np.sum(np.less(vec_true, vec_pred), axis=0)

    Metrics(BAD_METRIC=bad_metric)
    _, level, message = caplog.record_tuples[0]
    assert level == logging.WARNING
    assert re.match(r"^Metric \[BAD_METRIC\] does not meet", message) is not None

    with pytest.raises(ValueError, match=r"NON_CALL.*is not callable"):
        Metrics(NON_CALL=10)


def test_metrics_check_metric():
    return True
    # good_metrics = lambda x, y:
