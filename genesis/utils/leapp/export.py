"""Export complete tensor policy computations between public state and actuator boundaries."""

import inspect
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

import leapp
from leapp import Axis, TensorSemantics, annotate
from leapp.leapp_graph.datatypes.traced_data import TracedData

from genesis.utils.misc import tensor_to_array

from .annotations import get_boundary
from .types import (
    CallArgument,
    Connection,
    Controller,
    DeploymentMetadata,
    TensorMetadata,
    connection_to_dict,
    dof_names,
    propeller_names,
    resolve_dofs,
    save_metadata,
)


def _semantics(name, tensor, metadata, connection, *, is_batched):
    axes_trailing = []
    if metadata.element_names is not None:
        axes_trailing.append(Axis("element", names=list(metadata.element_names)))
    if metadata.component_names is not None:
        axes_trailing.append(Axis("component", names=list(metadata.component_names)))
    axes = [Axis("system")] if is_batched else []
    if len(axes) + len(axes_trailing) > tensor.ndim:
        raise ValueError(f"Semantic axes for '{name}' exceed its tensor rank.")
    axes.extend([None] * (tensor.ndim - len(axes) - len(axes_trailing)))
    axes.extend(axes_trailing)
    extra = {"genesis_connection": connection_to_dict(connection)}
    if metadata.unit is not None:
        extra["unit"] = metadata.unit
    return TensorSemantics(
        name=name,
        ref=tensor,
        kind=metadata.kind,
        axes=axes,
        is_setpoint=metadata.is_setpoint,
        reference=metadata.reference,
        expressed_in=metadata.expressed_in,
        extra=extra,
    )


def _named_values(names, values):
    if len(names) == 1:
        values = (values,)
    return dict(zip(names, values, strict=True))


def _selection_indices(selection, n_dofs):
    if isinstance(selection, TracedData):
        raise ValueError(
            "Degree-of-freedom bindings must use static indices. Gather dynamic selections inside the policy."
        )
    if selection is None:
        return list(range(n_dofs))
    if isinstance(selection, slice):
        return list(range(n_dofs))[selection]
    if isinstance(selection, torch.Tensor):
        selection = tensor_to_array(selection)
    if isinstance(selection, np.ndarray):
        selection = selection.tolist()
    if isinstance(selection, int):
        return [selection]
    if all(isinstance(value, bool) for value in selection):
        if len(selection) != n_dofs:
            raise ValueError("Degree-of-freedom selection mask has the wrong length.")
        return [i_d for i_d, is_selected in enumerate(selection) if is_selected]
    return list(selection)


class _EntityCapture:
    def __init__(self, entity_name, entity, node_name, outputs):
        self.entity_name = entity_name
        self.entity = entity
        self.node_name = node_name
        self.outputs = outputs
        self.calls = {}
        self.connections = []

    def __getattr__(self, name):
        boundary = get_boundary(self.entity, name)
        return partial(self._call, name, boundary)

    def _call(self, method_name, boundary, *args, **kwargs):
        method = getattr(self.entity, method_name)
        arguments = inspect.signature(method).bind(*args, **kwargs)
        arguments.apply_defaults()
        if arguments.arguments.get("envs_idx") is not None:
            raise ValueError("Export a complete fixed batch. Environment subset selection requires a separate binding.")
        metadata = boundary.metadata
        names = None
        if boundary.selection == "dof":
            selection = arguments.arguments.get("dofs_idx_local")
            indices = _selection_indices(selection, self.entity.n_dofs)
            names_available = dof_names(self.entity)
            names = tuple(names_available[i_d] for i_d in indices)
            resolve_dofs(self.entity, names)
            metadata = replace(metadata, element_names=names)
        elif boundary.selection == "propeller":
            metadata = replace(metadata, element_names=propeller_names(self.entity))
        arguments_saved = tuple(
            CallArgument(name, value)
            for name, value in arguments.arguments.items()
            if name not in (boundary.value_parameter, "dofs_idx_local", "envs_idx")
        )
        if any(not isinstance(argument.value, (bool, int, float, str, type(None))) for argument in arguments_saved):
            raise ValueError(f"'{method_name}' has a dynamic or unsupported boundary argument.")
        connection = Connection(
            entity_name=self.entity_name,
            method_name=method_name,
            arguments=arguments_saved,
            value_parameter=boundary.value_parameter,
            dof_names=names,
        )
        self.connections.append(connection)
        i_call = self.calls.get(method_name, 0)
        self.calls[method_name] = i_call + 1
        name = f"{self.entity_name}_{method_name}_{i_call}"
        if boundary.value_parameter is None:
            tensor = method(*args, **kwargs)
            semantics = _semantics(name, tensor, metadata, connection, is_batched=self.entity.scene.n_envs > 0)
            return annotate.input_tensors(self.node_name, [semantics])
        tensor = arguments.arguments[boundary.value_parameter]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Exported actuator targets must be Torch tensors computed from policy inputs or state.")
        self.outputs[name] = _semantics(name, tensor, metadata, connection, is_batched=self.entity.scene.n_envs > 0)
        return None


def _controllers(entities, connections):
    controllers = []
    for entity_name, entity in entities.items():
        names = tuple(
            dict.fromkeys(
                name
                for connection in connections
                if connection.entity_name == entity_name
                and connection.method_name in ("control_dofs_position", "control_dofs_velocity", "control_dofs_force")
                for name in connection.dof_names
            )
        )
        if not names:
            continue
        indices = resolve_dofs(entity, names)
        tensors = (
            entity.get_dofs_kp(indices),
            entity.get_dofs_kv(indices),
            *entity.get_dofs_force_range(indices),
        )
        values = []
        for tensor in tensors:
            if tensor.ndim == 2:
                if not torch.allclose(tensor, tensor[:1].expand_as(tensor)):
                    raise ValueError("Deployment requires identical controller settings across the captured batch.")
                tensor = tensor[0]
            values.append(tensor_to_array(tensor).tolist())
        kp, kv, force_lower, force_upper = values
        if any(math.isnan(value) for controller_values in values for value in controller_values):
            raise ValueError("Controller settings contain NaN.")
        controllers.append(
            Controller(
                entity_name=entity_name,
                dof_names=names,
                kp=kp,
                kv=kv,
                force_lower=[value if math.isfinite(value) else None for value in force_lower],
                force_upper=[value if math.isfinite(value) else None for value in force_upper],
            )
        )
    return tuple(controllers)


def export_policy(
    step: Callable,
    entities: Mapping,
    sample_inputs: Iterable[Mapping[str, torch.Tensor]],
    initial_state: Mapping[str, torch.Tensor],
    output_dir: str | Path,
    *,
    step_dt: float,
    name: str = "policy",
    input_semantics: Mapping[str, TensorMetadata] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Path:
    """Export preprocessing, policy inference and actuator conversion as one Open Neural Network Exchange model.

    ``step(entities, inputs, state)`` reads annotated entity getters, writes annotated control setters and returns all
    next-state tensors by name. During capture, entity adapters intercept those calls without applying control or
    stepping physics. All state is explicit and resets to ``initial_state``. Policy weights and other closed-over
    tensors are constants. Callbacks must use traceable tensor computation with fixed shapes and static control flow.

    Each item of ``sample_inputs`` supplies named external tensors, such as commands, for another capture step. A
    generator may update real entity state between samples. Supply representative changing samples. ``step_dt`` is
    an integer multiple of the scene timestep. ``input_semantics`` optionally describes external tensor meanings.

    The returned YAML path references the model and initial feedback values. A sibling ``.genesis.json`` describes
    deployment timing and controllers. Validation compares cached outputs and feedback updates at ``rtol``/``atol``.
    """
    if not entities:
        raise ValueError("At least one scene entity binding is required for Genesis policy export.")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("Export names must start with a letter and contain only letters, digits and underscores.")
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) for key in entities):
        raise ValueError("Entity binding names must contain only letters, digits and underscores.")
    scene = next(iter(entities.values())).scene
    if any(entity.scene is not scene for entity in entities.values()):
        raise ValueError("Captured entities must belong to the same scene.")
    ratio = step_dt / scene.dt
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio)):
        raise ValueError("Policy timestep must be a positive integer multiple of the scene timestep.")
    input_semantics = {} if input_semantics is None else input_semantics
    state = {key: value.clone() for key, value in initial_state.items()}
    connections = []
    n_samples = 0
    node_name = "policy"
    leapp.start(name=name, save_path=str(output_dir), max_cached_io=5)
    try:
        with torch.no_grad():
            for inputs in sample_inputs:
                if any(not isinstance(value, torch.Tensor) for value in inputs.values()):
                    raise TypeError("External policy inputs must be named Torch tensors.")
                annotations = {
                    f"external_{key}": _semantics(
                        f"external_{key}",
                        value,
                        input_semantics.get(key, TensorMetadata()),
                        Connection(external_name=key),
                        is_batched=scene.n_envs > 0,
                    )
                    for key, value in inputs.items()
                }
                tensors = annotate.input_tensors(node_name, list(annotations.values()) if annotations else {})
                inputs_traced = _named_values(inputs, tensors)
                if state:
                    tensors = annotate.state_tensors(node_name, {f"state_{key}": value for key, value in state.items()})
                    state_traced = _named_values(state, tensors)
                else:
                    state_traced = {}
                outputs = {}
                adapters = {key: _EntityCapture(key, entity, node_name, outputs) for key, entity in entities.items()}
                state_next = step(adapters, inputs_traced, state_traced)
                if set(state_next) != set(initial_state):
                    raise ValueError("Inference step must return exactly the declared feedback-state names.")
                if not outputs:
                    raise ValueError("Inference step produced no annotated actuator targets.")
                if state:
                    annotate.update_state(node_name, {f"state_{key}": state_next[key] for key in state})
                annotate.output_tensors(node_name, list(outputs.values()), export_with="onnx")
                state = {key: value.detach().clone() for key, value in state_next.items()}
                connections_current = [
                    connection for adapter in adapters.values() for connection in adapter.connections
                ]
                if n_samples and connections_current != connections:
                    raise ValueError("Entity boundary calls changed between capture samples.")
                connections = connections_current
                n_samples += 1
    finally:
        leapp.stop()
    if n_samples == 0:
        raise ValueError("At least one capture sample is required.")
    leapp.compile_graph(visualize=False, validate=True, strict=True, rtol=rtol, atol=atol)
    bundle = Path(output_dir) / name / f"{name}.yaml"
    metadata = DeploymentMetadata(physics_dt=scene.dt, step_dt=step_dt, controllers=_controllers(entities, connections))
    save_metadata(bundle, metadata)
    return bundle
