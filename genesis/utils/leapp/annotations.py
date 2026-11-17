"""Metadata-only annotations for physical policy input and output boundaries."""

import inspect
from dataclasses import dataclass

from .types import TensorMetadata


@dataclass(frozen=True)
class Boundary:
    metadata: TensorMetadata
    value_parameter: str | None = None
    selection: str | None = None


_BOUNDARIES = {}


def tensor_input(*, kind=None, component_names=None, unit=None, expressed_in=None, selection=None):
    """Register a state getter, preserving the original callable and its execution."""
    boundary = Boundary(
        metadata=TensorMetadata(kind=kind, component_names=component_names, unit=unit, expressed_in=expressed_in),
        selection=selection,
    )

    def decorate(function):
        _BOUNDARIES[function] = boundary
        return function

    return decorate


def tensor_output(*, value_parameter, kind=None, unit=None, selection=None):
    """Register the physical target argument of a control setter, preserving its execution."""
    boundary = Boundary(
        metadata=TensorMetadata(kind=kind, unit=unit),
        value_parameter=value_parameter,
        selection=selection,
    )

    def decorate(function):
        _BOUNDARIES[function] = boundary
        return function

    return decorate


def get_boundary(entity, method_name: str) -> Boundary:
    """Resolve registered metadata through ordinary method wrappers.

    Reflection is confined to this extensible boundary registry. Unannotated attributes cannot become frozen inputs.
    """
    function = inspect.unwrap(inspect.getattr_static(type(entity), method_name))
    try:
        return _BOUNDARIES[function]
    except KeyError:
        raise ValueError(f"{type(entity).__name__}.{method_name} has no policy export boundary annotation.") from None
