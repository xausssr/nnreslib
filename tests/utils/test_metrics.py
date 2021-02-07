import logging
import re

import numpy as np
import pytest

from nnreslib.utils.metrics import MetricFlags, Metrics
from nnreslib.utils.metrics.metrics import STANDART_METRICS
from nnreslib.utils.metrics.regression_metrics import _rmse


def test_init(caplog):
    Metrics(MY_RMSE=_rmse)
    assert not caplog.records

    def bad_metric(vec_true, vec_pred):
        result = 0.0
        for out_true, out_pred in zip(vec_true, vec_pred):
            result += np.sum(np.less(out_true, out_pred))
        return result / len(vec_true)

    Metrics(BAD_METRIC=bad_metric)
    _, level, message = caplog.record_tuples[0]
    assert level == logging.WARNING
    assert re.match(r"^Metric \[BAD_METRIC\] does not meet", message) is not None

    with pytest.raises(ValueError, match=r"NON_CALL.*is not callable"):
        Metrics(NON_CALL=10)


def test_metric_flags():
    for metric_name in STANDART_METRICS:
        MetricFlags[metric_name]  # pylint:disable=pointless-statement


def test_check_metric():
    return True
    # good_metrics = lambda x, y:
