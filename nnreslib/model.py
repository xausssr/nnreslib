from __future__ import annotations

import collections.abc as ca
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Generator, Iterable, List, Mapping, Sequence, Tuple, Union, cast

from .layers import ConvolutionLayer, FlattenLayer, FullyConnectedLayer, InputLayer, Layer, MaxPoolLayer, TrainableLayer

if TYPE_CHECKING:
    from .utils.types import Shape

ArchitectureLevelKeyType = Union[str, Sequence[str]]
ArchitectureLevelValueType = Union[Layer, Sequence[Layer]]
ArchitectureLevelType = Union[ArchitectureLevelValueType, Mapping[ArchitectureLevelKeyType, ArchitectureLevelValueType]]
ArchitectureType = Sequence[ArchitectureLevelType]

ParseType = Tuple[Sequence[str], Layer]
ParseResultType = Generator[ParseType, None, None]


@dataclass
class LayerInfo:
    layer_id: int
    layer: Layer


class Model:
    __slots__ = (
        "neurons_count",
        "_layers",
        "_input_layers",
        "_output_layers",
        "_initialized_layers",
        "_architecture",
    )

    def __init__(self, architecture: ArchitectureType):
        self.neurons_count = 0
        self._layers: Dict[str, LayerInfo] = {}
        self._input_layers: List[str] = []
        self._output_layers: List[str] = []
        self._initialized_layers: List[str] = []
        self._architecture: Dict[str, Sequence[str]] = {}
        self._parse_layers(architecture)
        self._build_model(architecture)

    # TODO: load architecture from json/yaml

    @staticmethod
    def _parse_layer_plain_definition(level: ArchitectureLevelValueType) -> ParseResultType:
        if isinstance(level, Layer):
            yield (("",), level)
        else:
            for layer in level:
                yield (("",), layer)

    @staticmethod
    def _parse_layer_dict_definition(
        level: Mapping[ArchitectureLevelKeyType, ArchitectureLevelValueType]
    ) -> ParseResultType:
        def fix_input_layers(input_layers: ArchitectureLevelKeyType) -> Sequence[str]:
            if isinstance(input_layers, str):
                return (input_layers,)
            return input_layers

        for _input_layers, layers in level.items():
            input_layers = fix_input_layers(_input_layers)
            for _, layer in Model._parse_layer_plain_definition(layers):
                yield (input_layers, layer)

    @staticmethod
    def _parse_layer_definition(level: ArchitectureLevelType) -> ParseResultType:
        if isinstance(level, ca.Mapping):
            for layer in Model._parse_layer_dict_definition(level):
                yield layer
        else:
            for layer in Model._parse_layer_plain_definition(level):
                yield layer

    @staticmethod
    def _check_first_layer(first_layers: ArchitectureLevelType) -> Sequence[str]:
        if isinstance(first_layers, ca.Mapping):
            raise ValueError("First layers must be single layer or sequence of layers, not Mapping")
        parsed_layers = list(Model._parse_layer_plain_definition(first_layers))
        if not all(
            map(
                lambda x: x[0] == ("",) and isinstance(x[1], InputLayer),
                parsed_layers,
            )
        ):
            raise ValueError("First layer(s) must be InputLayer")

        return tuple(x[1].name for x in parsed_layers)

    def _parse_layers(self, archiecture: ArchitectureType) -> None:
        layer_id = 1
        for level in archiecture:
            for _, layer in Model._parse_layer_definition(level):
                if layer.name in self._layers:
                    raise ValueError(f"Layer's name must be unique: {layer.name}!")
                self._layers[layer.name] = LayerInfo(layer_id, layer)
                layer_id += 1
                if isinstance(layer, InputLayer):
                    self._input_layers.append(layer.name)
                elif isinstance(layer, TrainableLayer):
                    self._initialized_layers.append(layer.name)
                if layer.is_out:
                    self._output_layers.append(layer.name)

    def _check_input_layers_type(self, input_layers: Iterable[str], layer: Layer) -> None:
        pre_layers_info = [self._layers[x] for x in input_layers]
        pre_layers = [x.layer for x in pre_layers_info]
        if not (
            all(map(lambda x: isinstance(x, InputLayer), pre_layers))
            or all(map(lambda x: isinstance(x, (ConvolutionLayer, MaxPoolLayer)), pre_layers))
            and isinstance(layer, (ConvolutionLayer, MaxPoolLayer, FlattenLayer))
            or all(map(lambda x: isinstance(x, (FullyConnectedLayer, FlattenLayer)), pre_layers))
            and isinstance(layer, FullyConnectedLayer)
        ):
            raise ValueError(f"Unsupported layer sequence: {','.join(input_layers)} -> {layer.name}")

        layer_id = self._layers[layer.name].layer_id
        if all(map(lambda x: x > layer_id, (x.layer_id for x in pre_layers_info))):
            raise ValueError(f"You must use at least one non feedback layer as input layer for layer: {layer.name}")

    def _get_layer_inputs_shapes(self, layer: Layer, input_layers: Sequence[str]) -> Tuple[Shape, Iterable[Shape]]:
        if layer.merge is None:
            if len(input_layers) > 1:
                raise ValueError(f"For multiple inputs you need to specify a 'merge' for layer {layer.name}")
            return self._layers[input_layers[0]].layer.output_shape, ()

        if not layer.merge.main_input:
            layer.merge.main_input = input_layers[0]
            input_layers = input_layers[1:]
        elif layer.merge.main_input not in input_layers:
            raise ValueError(f"Layer '{layer.name}' doesn't depend on main input '{layer.merge.main_input}'")

        return (
            self._layers[layer.merge.main_input].layer.output_shape,
            (self._layers[x].layer.output_shape for x in input_layers),
        )

    def _process_layer(self, input_layers: Sequence[str], layer: Layer) -> None:
        self._check_input_layers_type(input_layers, layer)
        main_input, others_inputs = self._get_layer_inputs_shapes(layer, input_layers)
        layer.set_shapes(main_input, *others_inputs)

        self.neurons_count += layer.neurons_count

    def _build_model(self, architecture: ArchitectureType) -> None:
        pre_level = Model._check_first_layer(architecture[0])

        for level in architecture[1:]:
            new_pre_level: List[str] = []
            for dependencies, layer in Model._parse_layer_definition(level):
                new_pre_level.append(layer.name)
                if "" in dependencies:
                    dependencies = pre_level  # TODO: check dependencies layers compatibility with merge function
                self._process_layer(dependencies, layer)
                self._architecture[layer.name] = dependencies
            pre_level = new_pre_level

    def initialize(self, data_mean: float = 0.0, data_std: float = 0.0) -> None:
        for layer_name in self._initialized_layers:
            layer: TrainableLayer = cast(TrainableLayer, self._layers[layer_name].layer)
            layer.set_weights(data_mean, data_std)
            layer.set_biases(data_mean, data_std)

    @property
    def layers(self) -> Generator[Layer, None, None]:
        for layer_info in self._layers.values():
            yield layer_info.layer

    @property
    def input_layers(self) -> Generator[InputLayer, None, None]:
        for input_layer in self._input_layers:
            yield cast(InputLayer, self._layers[input_layer].layer)

    @property
    def output_layers(self) -> Generator[Layer, None, None]:
        for output_layer in self._output_layers:
            yield self._layers[output_layer].layer

    @property
    def architecture(self) -> Generator[Tuple[Layer, Sequence[str]], None, None]:
        for layer, inputs_layers in self._architecture.items():
            yield self._layers[layer].layer, inputs_layers
