import numpy as np

from nnreslib.utils.metrics.categorial_metrics import CalcCategorialMetrics, CategorialMetrics


def test_cat_metrics_add():
    cat_1 = CategorialMetrics(1, 2, 3, 4)
    cat_2 = CategorialMetrics(10, 20, 30, 40)

    sum_cat = cat_1 + cat_2
    assert sum_cat.TP == 11
    assert sum_cat.FP == 22
    assert sum_cat.FN == 33
    assert sum_cat.TN == 44

    cat_1 += cat_2
    assert sum_cat.TP == cat_1.TP
    assert sum_cat.FP == cat_1.FP
    assert sum_cat.FN == cat_1.FN
    assert sum_cat.TN == cat_1.TN


def test_calc_cat_metrics_add():
    cat_1 = CalcCategorialMetrics([np.array([0, 1, 0, 1])], [np.array([0.7, 0.7, 0.2, 0.3])])
    cat_2 = CalcCategorialMetrics([np.array([1, 0, 1, 0])], [np.array([0.4, 0.7, 0.7, 0.3])])

    # pylint:disable=protected-access
    sum_cat = cat_1 + cat_2
    assert sum_cat._metrics[0].TP == 2
    assert sum_cat._metrics[0].FP == 2
    assert sum_cat._metrics[0].FN == 2
    assert sum_cat._metrics[0].TN == 2

    cat_1 += cat_2
    assert sum_cat._metrics[0].TP == cat_1._metrics[0].TP
    assert sum_cat._metrics[0].FP == cat_1._metrics[0].FP
    assert sum_cat._metrics[0].FN == cat_1._metrics[0].FN
    assert sum_cat._metrics[0].TN == cat_1._metrics[0].TN
    # pylint:enable=protected-access
