from nnreslib.graph.fit_graph import FitGraph, FitMethods


def test_fit_methods():
    for method in FitMethods:
        FitGraph.get_fitter(method)

    for method in FitGraph._fitters:  # pylint:disable=protected-access
        FitMethods[method]  # pylint:disable=pointless-statement
