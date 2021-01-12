from __future__ import annotations

from abc import ABC
from typing import Callable, Dict, Type

from .. import ForwardGraph
from ...architecture import Architecture
from ...backend import graph as G


class FitGraph(ABC):
    _fitters: Dict[str, Type[FitGraph]] = {}

    __slots__ = ("outputs", "model_outputs")

    def __init__(
        self,
        batch_size: int,
        architecture: Architecture,
        forward_graph: ForwardGraph,  # pylint: disable=unused-argument
    ) -> None:
        self.outputs = [
            G.placeholder(name=x.name, shape=(batch_size, *x.output_shape)) for x in architecture.output_layers
        ]

        # current_node = last output layer
        last_node = None  # FIXME: get correct last graph node
        # XXX multiple outputs must be in one tensor like: G.concat(G.flatten(out_1), ..., G.flatten(out_n))
        self.model_outputs = G.squeeze(last_node)  # or replace by correct G.flatten ^^^^^

    @classmethod
    def register_fitter(cls) -> Callable[[Type[FitGraph]], Type[FitGraph]]:
        def decorator(fitter: Type[FitGraph]) -> Type[FitGraph]:
            cls._fitters[fitter.__name__] = fitter
            return fitter

        return decorator

    # XXX: implement call fit on fitter directly from this method (get_fitter -> fit)
    @classmethod
    def get_fitter(cls, fit_type: str) -> Type[FitGraph]:
        fitter = cls._fitters.get(fit_type)
        if fitter:
            return fitter
        raise ValueError(f"Unsupported fit type: {fit_type}")
