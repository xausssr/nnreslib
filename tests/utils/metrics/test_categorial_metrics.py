from statistics import mean

import numpy as np

from nnreslib.utils.metrics.categorial_metrics import (
    CalcCategorialMetrics,
    CategorialMetrics,
    CategorialMetricsAggregation,
)


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


def test_cat_metrics_aggregate():
    cat_1 = CategorialMetrics(1, 2, 3, 4, Recall=123)
    cat_2 = CategorialMetrics(10, 20, 30, 40, Recall=321)

    cat_mean = CategorialMetrics.aggregate(cat_1, cat_2, CategorialMetricsAggregation.MEAN)
    assert cat_mean.TP == mean((cat_1.TP, cat_2.TP))
    assert cat_mean.FP == mean((cat_1.FP, cat_2.FP))
    assert cat_mean.FN == mean((cat_1.FN, cat_2.FN))
    assert cat_mean.TN == mean((cat_1.TN, cat_2.TN))
    assert cat_mean.Recall == mean((cat_1.Recall, cat_2.Recall))

    cat_max = CategorialMetrics.aggregate(cat_1, cat_2, CategorialMetricsAggregation.MAX)
    assert cat_max.TP == max((cat_1.TP, cat_2.TP))
    assert cat_max.FP == max((cat_1.FP, cat_2.FP))
    assert cat_max.FN == max((cat_1.FN, cat_2.FN))
    assert cat_max.TN == max((cat_1.TN, cat_2.TN))
    assert cat_max.Recall == max((cat_1.Recall, cat_2.Recall))

    cat_min = CategorialMetrics.aggregate(cat_1, cat_2, CategorialMetricsAggregation.MIN)
    assert cat_min.TP == min((cat_1.TP, cat_2.TP))
    assert cat_min.FP == min((cat_1.FP, cat_2.FP))
    assert cat_min.FN == min((cat_1.FN, cat_2.FN))
    assert cat_min.TN == min((cat_1.TN, cat_2.TN))
    assert cat_min.Recall == min((cat_1.Recall, cat_2.Recall))


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


def test_calc_cat_metrics_get_metrics_without_aggregation():
    cat_1 = CalcCategorialMetrics([np.array([0, 1, 0, 1])], [np.array([0.7, 0.7, 0.2, 0.3])])
    cat_2 = CalcCategorialMetrics([np.array([1, 0, 1, 0])], [np.array([0.4, 0.7, 0.7, 0.3])])
    sum_cat = cat_1 + cat_2
    result = sum_cat.get_metrics()
    assert result.metrics == sum_cat._metrics  # pylint:disable=protected-access
    assert result.aggregate is None


def test_calc_cat_metrics_get_metrics_with_aggregation():
    # One out
    cat = CalcCategorialMetrics(
        [np.array([0, 1, 0, 1])], [np.array([0.7, 0.7, 0.2, 0.3])], aggregation=CategorialMetricsAggregation.MAX
    )
    result = cat.get_metrics()
    cat_c = cat._metrics.copy()  # pylint:disable=protected-access
    cat_c[0].calc_metrics()
    assert result.metrics == cat_c
    assert result.aggregate is not None
    assert result.aggregate == cat_c[0]

    # Two outs
    cat_2 = CalcCategorialMetrics(
        [np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])],
        [np.array([0.7, 0.7, 0.2, 0.3]), np.array([0.4, 0.7, 0.7, 0.3])],
        aggregation=CategorialMetricsAggregation.MAX,
    )
    result = cat_2.get_metrics()
    cat_2_c = cat_2._metrics.copy()  # pylint:disable=protected-access
    cat_2_c[0].calc_metrics()
    cat_2_c[1].calc_metrics()
    assert result.metrics == cat_2_c
    assert result.aggregate is not None
    assert result.aggregate == CategorialMetrics.aggregate(
        *cat_2._metrics,  # pylint:disable=protected-access
        agg_func=CategorialMetricsAggregation.MAX,
    )
