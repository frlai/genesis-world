"""Run exported policy pipelines against a Genesis scene."""

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from leapp import InferenceManager
from leapp.utils.tensor_description import map_to_torch_dtype

import genesis as gs
from genesis.utils.leapp.annotations import get_boundary
from genesis.utils.leapp.types import connection_from_dict, load_metadata, propeller_names, resolve_dofs


@dataclass(frozen=True)
class _InputBinding:
    name: str
    external_name: str | None
    read: Callable | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _OutputBinding:
    name: str
    write: Callable
    value_parameter: str


class LeappDeployment:
    """Run a complete exported policy pipeline at its recorded control period.

    Parameters
    ----------
    scene : genesis.Scene
        Built scene whose timestep matches the export.
    bundle : str | pathlib.Path
        Pipeline YAML file with its adjacent Genesis metadata file.
    entities : Mapping[str, RigidEntity]
        Entities keyed by the names supplied during export.
    reset_callback : Callable[[], None] | None, optional
        Restore the scene and its initial control state. If None, use ``scene.reset``. Defaults to None.

    The caller owns the scene. Inputs use the external names supplied to the exporter. Each call to ``step`` advances
    one control period, holding actuator targets over the corresponding physics steps. ``reset`` restores the scene
    and exported feedback initialization together.
    """

    def __init__(self, scene, bundle, entities, *, reset_callback=None):
        if not scene.is_built:
            raise ValueError("Build the deployment scene before loading a policy.")
        bundle = Path(bundle)
        metadata = load_metadata(bundle)
        if not math.isfinite(metadata.physics_dt) or metadata.physics_dt <= 0:
            raise ValueError("The exported physics timestep must be finite and positive.")
        if not math.isfinite(metadata.step_dt) or metadata.step_dt <= 0:
            raise ValueError("The exported control period must be finite and positive.")
        if not math.isclose(scene.dt, metadata.physics_dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Scene timestep {scene.dt} differs from exported timestep {metadata.physics_dt}.")
        n_steps = round(metadata.step_dt / scene.dt)
        if n_steps < 1 or not math.isclose(n_steps * scene.dt, metadata.step_dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("The exported control period must be a positive integer multiple of the scene timestep.")
        for entity in entities.values():
            if entity.scene is not scene:
                raise ValueError("Every deployment entity must belong to the supplied scene.")

        self.scene = scene
        self.inference = InferenceManager(str(bundle))
        self._reset_callback = scene.reset if reset_callback is None else reset_callback
        self._n_steps = n_steps
        self._is_closed = False
        self._inputs = []
        self._outputs = []
        for port_name in self.inference.inputs:
            node_name, tensor_name = port_name.split("/")
            node = self.inference.nodes[node_name]
            description = next(item for item in node.input_descriptions if item["name"] == tensor_name)
            if "genesis_connection" not in description:
                raise ValueError(f"Input '{port_name}' has no Genesis connection metadata.")
            connection = connection_from_dict(description["genesis_connection"])
            reader = None
            if connection.external_name is None:
                if connection.entity_name not in entities:
                    raise ValueError(f"Input '{port_name}' requires entity '{connection.entity_name}'.")
                entity = entities[connection.entity_name]
                boundary = get_boundary(entity, connection.method_name)
                if boundary.value_parameter is not None or connection.value_parameter is not None:
                    raise ValueError(f"Input '{port_name}' must bind a state getter.")
                arguments = {argument.name: argument.value for argument in connection.arguments}
                if connection.dof_names is not None:
                    arguments["dofs_idx_local"] = resolve_dofs(entity, connection.dof_names)
                reader = partial(getattr(entity, connection.method_name), **arguments)
            elif connection.entity_name is not None or connection.method_name is not None:
                raise ValueError(f"Input '{port_name}' has both external and entity bindings.")
            shape = description["shape"]
            if isinstance(shape, str):
                shape = json.loads(shape)
            self._inputs.append(
                _InputBinding(
                    port_name,
                    connection.external_name,
                    reader,
                    tuple(shape),
                    map_to_torch_dtype(description["dtype"]),
                    node.device,
                )
            )

        for port_name in self.inference.outputs:
            node_name, tensor_name = port_name.split("/")
            node = self.inference.nodes[node_name]
            description = next(item for item in node.output_descriptions if item["name"] == tensor_name)
            if "genesis_connection" not in description:
                raise ValueError(f"Output '{port_name}' has no Genesis connection metadata.")
            connection = connection_from_dict(description["genesis_connection"])
            if connection.entity_name not in entities:
                raise ValueError(f"Output '{port_name}' requires entity '{connection.entity_name}'.")
            entity = entities[connection.entity_name]
            boundary = get_boundary(entity, connection.method_name)
            if boundary.value_parameter is None or connection.value_parameter != boundary.value_parameter:
                raise ValueError(f"Output '{port_name}' must bind the annotated actuator value parameter.")
            arguments = {argument.name: argument.value for argument in connection.arguments}
            if connection.dof_names is not None:
                arguments["dofs_idx_local"] = resolve_dofs(entity, connection.dof_names)
            if boundary.selection == "propeller":
                axes = description.get("axes", [])
                names = next((axis["names"] for axis in axes if axis and axis.get("kind") == "element"), None)
                if names is None or tuple(names) != propeller_names(entity):
                    raise ValueError(f"Output '{port_name}' propeller order differs from the deployment entity.")
            writer = partial(getattr(entity, connection.method_name), **arguments)
            self._outputs.append(_OutputBinding(port_name, writer, connection.value_parameter))

        self._external_names = {binding.external_name for binding in self._inputs if binding.external_name is not None}
        for controller in metadata.controllers:
            if controller.entity_name not in entities:
                raise ValueError(f"Controller requires entity '{controller.entity_name}'.")
            entity = entities[controller.entity_name]
            dofs_idx_local = resolve_dofs(entity, controller.dof_names)
            entity.set_dofs_kp(controller.kp, dofs_idx_local=dofs_idx_local)
            entity.set_dofs_kv(controller.kv, dofs_idx_local=dofs_idx_local)
            force_lower = [float("-inf") if value is None else value for value in controller.force_lower]
            force_upper = [float("inf") if value is None else value for value in controller.force_upper]
            entity.set_dofs_force_range(force_lower, force_upper, dofs_idx_local=dofs_idx_local)

    def reset(self):
        """Restore the scene and the exported initial feedback tensors."""
        if self._is_closed:
            raise RuntimeError("The deployment runner is closed.")
        self._reset_callback()
        self.inference.reset()

    def read_inputs(self, external_inputs: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Read raw scene and external tensors using the exported input connections.

        ``external_inputs`` supplies every exported external input by name, matching the exported shapes and dtypes.
        The returned mapping can be passed directly to ``inference.run_policy`` for evaluation without stepping.
        """
        if self._is_closed:
            raise RuntimeError("The deployment runner is closed.")
        external_inputs = {} if external_inputs is None else external_inputs
        if external_inputs.keys() != self._external_names:
            raise ValueError(
                f"External input names must be {sorted(self._external_names)}, got {sorted(external_inputs)}."
            )
        inputs = {}
        for binding in self._inputs:
            value = binding.read() if binding.external_name is None else external_inputs[binding.external_name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Input '{binding.name}' must be a torch tensor.")
            if tuple(value.shape) != binding.shape or value.dtype != binding.dtype:
                raise ValueError(
                    f"Input '{binding.name}' expects shape {binding.shape} and dtype {binding.dtype}, "
                    f"got {tuple(value.shape)} and {value.dtype}."
                )
            # InferenceManager retains inputs and zeros them on reset. Own the buffers so borrowed scene views stay intact.
            inputs[binding.name] = value.to(device=binding.device, copy=True)
        return inputs

    def step(self, external_inputs: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Run one policy evaluation and advance the scene by its exported control period.

        ``external_inputs`` uses the same names and tensor contracts as ``read_inputs``. Returned tensors are actuator
        targets keyed by their pipeline output names.
        """
        inputs = self.read_inputs(external_inputs)
        with torch.no_grad():
            outputs = self.inference.run_policy(inputs)
        targets = {name: value.to(device=gs.device) for name, value in outputs.items()}
        for _ in range(self._n_steps):
            for binding in self._outputs:
                binding.write(**{binding.value_parameter: targets[binding.name]})
            self.scene.step()
        return outputs

    def close(self):
        """Release policy resources while leaving the supplied scene under caller ownership."""
        self._inputs.clear()
        self._outputs.clear()
        self.inference = None
        self._is_closed = True
