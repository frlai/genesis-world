"""Portable connection descriptions shared by policy export and deployment."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorMetadata:
    """Describe a tensor's quantity and ordered final axes without importing an export backend.

    ``element_names`` and ``component_names`` describe successive trailing axes. Any preceding axis is the system
    axis. ``reference=None`` denotes the world frame. Units describe the stored values without converting them.
    """

    kind: str | None = None
    element_names: tuple[str, ...] | None = None
    component_names: tuple[str, ...] | None = None
    reference: str | None = None
    expressed_in: str | None = None
    unit: str | None = None
    is_setpoint: bool | None = None


@dataclass(frozen=True)
class CallArgument:
    name: str
    value: bool | int | float | str | list[int] | None


@dataclass(frozen=True)
class Connection:
    entity_name: str | None = None
    method_name: str | None = None
    arguments: tuple[CallArgument, ...] = ()
    value_parameter: str | None = None
    dof_names: tuple[str, ...] | None = None
    external_name: str | None = None


@dataclass(frozen=True)
class Controller:
    entity_name: str
    dof_names: tuple[str, ...]
    kp: list[float]
    kv: list[float]
    force_lower: list[float | None]
    force_upper: list[float | None]


@dataclass(frozen=True)
class DeploymentMetadata:
    physics_dt: float
    step_dt: float
    controllers: tuple[Controller, ...] = ()
    schema_version: int = 1


def dof_names(entity) -> tuple[str, ...]:
    """Name each degree of freedom in entity-local order, including components of multi-axis joints."""
    return tuple(
        joint.name if joint.n_dofs == 1 else f"{joint.name}/{i_d}"
        for joint in entity.joints
        for i_d in range(joint.n_dofs)
    )


def resolve_dofs(entity, names) -> list[int]:
    """Resolve ordered degree-of-freedom names against a deployment entity."""
    names_available = dof_names(entity)
    if len(set(names_available)) != len(names_available):
        raise ValueError("Entity degree-of-freedom names must be unique for policy deployment.")
    if len(set(names)) != len(names):
        raise ValueError("Selected degree-of-freedom names must be unique.")
    return [names_available.index(name) for name in names]


def propeller_names(entity) -> tuple[str, ...]:
    """Return propeller link names in the actuator's declared order."""
    return tuple(entity.morph.propellers_link_name)


def connection_to_dict(connection: Connection) -> dict:
    payload = asdict(connection)
    payload["arguments"] = list(payload["arguments"])
    if connection.dof_names is not None:
        payload["dof_names"] = list(connection.dof_names)
    return payload


def connection_from_dict(payload) -> Connection:
    return Connection(
        entity_name=payload["entity_name"],
        method_name=payload["method_name"],
        arguments=tuple(CallArgument(**argument) for argument in payload["arguments"]),
        value_parameter=payload["value_parameter"],
        dof_names=None if payload["dof_names"] is None else tuple(payload["dof_names"]),
        external_name=payload["external_name"],
    )


def load_metadata(bundle: str | Path) -> DeploymentMetadata:
    """Read the Genesis deployment description beside a pipeline's YAML file."""
    with Path(bundle).with_suffix(".genesis.json").open() as stream:
        payload = json.load(stream)
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported Genesis deployment metadata version.")
    return DeploymentMetadata(
        physics_dt=payload["physics_dt"],
        step_dt=payload["step_dt"],
        controllers=tuple(
            Controller(**{**controller, "dof_names": tuple(controller["dof_names"])})
            for controller in payload["controllers"]
        ),
    )


def save_metadata(bundle: str | Path, metadata: DeploymentMetadata) -> None:
    with Path(bundle).with_suffix(".genesis.json").open("w") as stream:
        json.dump(asdict(metadata), stream, indent=2, allow_nan=False)
        stream.write("\n")
