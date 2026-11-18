"""Metadata-only annotations for physical policy input and output boundaries."""

import inspect
from dataclasses import dataclass

from genesis.utils.misc import _BuildGuard

from .types import EnvironmentInput, TensorMetadata


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
        _BOUNDARIES[function.member if isinstance(function, _BuildGuard) else function] = boundary
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
        _BOUNDARIES[function.member if isinstance(function, _BuildGuard) else function] = boundary
        return function

    return decorate


def get_boundary(entity, method_name: str) -> Boundary:
    """Resolve registered metadata through ordinary method wrappers.

    Reflection is confined to this extensible boundary registry. Unannotated attributes cannot become frozen inputs.
    """
    function = inspect.unwrap(inspect.getattr_static(type(entity), method_name), stop=_BOUNDARIES.__contains__)
    try:
        return _BOUNDARIES[function]
    except KeyError:
        raise ValueError(f"{type(entity).__name__}.{method_name} has no policy export boundary annotation.") from None


@dataclass(frozen=True)
class PolicyEnvironment:
    entities: tuple[str, ...]
    inputs: tuple[EnvironmentInput, ...]
    state: tuple[str, ...]
    action: str
    skip: tuple[str, ...]
    step_dt: str


_ENVIRONMENTS = {}


def policy_environment(*, entities, inputs=(), state=(), action="actions", skip=(), step_dt="dt"):
    """Describe native environment capture on ``get_observations`` without changing execution.

    ``action`` names the tensor attribute supplying the action shape. ``skip`` lists training-only methods, such as
    command resampling and episode reset, disabled only on export-owned snapshots. The original ``step`` method must
    call ``scene.step`` exactly once and return observations first; capture interposes that physics boundary.
    """
    metadata = PolicyEnvironment(tuple(entities), tuple(inputs), tuple(state), action, tuple(skip), step_dt)

    def decorate(function):
        _ENVIRONMENTS[function] = metadata
        return function

    return decorate


def get_policy_environment(env) -> PolicyEnvironment:
    try:
        function = inspect.getattr_static(type(env), "get_observations")
    except AttributeError:
        raise ValueError("Environment must define an annotated get_observations method.") from None
    function = inspect.unwrap(function, stop=_ENVIRONMENTS.__contains__)
    try:
        return _ENVIRONMENTS[function]
    except KeyError:
        raise ValueError("Environment get_observations must have a policy_environment annotation.") from None
