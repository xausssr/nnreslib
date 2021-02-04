import logging
import re

import numpy as np
import pytest

from nnreslib.utils.metrics import Metrics, _mse


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
