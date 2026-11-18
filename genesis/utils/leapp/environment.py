"""Capture annotated environment methods without duplicating their inference computations."""

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, MethodType
from enum import Enum
from functools import partial

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from .annotations import get_boundary, get_policy_environment
from .export import export_policy


class _PreambleGuard(TorchDispatchMode):
    """Reject values escaping the discarded pre-physics computation into policy inference."""

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.tensors = {}
        self.derived = set()
        self.is_injecting = False
        self.is_reading = False

    def track(self, values, *, is_derived=True):
        for value in tree_leaves(values):
            if isinstance(value, torch.Tensor):
                self.tensors[id(value)] = value
                if is_derived:
                    self.derived.add(id(value))

    def check(self, values):
        if any(id(value) in self.tensors for value in tree_leaves(values) if isinstance(value, torch.Tensor)):
            raise ValueError(
                "A tensor from before scene.step is used after the capture boundary. Read state and compute policy "
                "observations after scene.step; pre-physics local values cannot be exported as constants."
            )

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if self.is_reading:
            return function(*args, **kwargs)
        values = (args, kwargs)
        if self.scene.is_observing and not self.is_injecting:
            self.check(values)
        elif not self.is_injecting and function == torch.ops.aten._local_scalar_dense.default:
            self.check(values)
        result = function(*args, **kwargs)
        if not self.scene.is_observing or (
            self.is_injecting and any(id(value) in self.derived for value in tree_leaves(values))
        ):
            self.track(result)
        return result


class _CaptureComplete(Exception):
    pass


class _UnavailableObject:
    def __init__(self, type_name):
        self.type_name = type_name

    def __getattr__(self, name):
        raise ValueError(f"{self.type_name}.{name} is not a declared environment export boundary.")


class _CaptureScene:
    def __init__(self, scene, begin_observation):
        self.scene = scene
        self.begin_observation = begin_observation
        self.is_observing = False

    def step(self, *args, **kwargs):
        if self.is_observing:
            raise _CaptureComplete()
        self.is_observing = True
        self.begin_observation()

    def get_error_envs_mask(self):
        return self.scene.get_error_envs_mask().clone()

    def __getattr__(self, name):
        raise ValueError(f"Scene.{name} is not supported during environment export.")


class _CaptureEntity:
    def __init__(self, entity, adapter, scene):
        self.entity = entity
        self.adapter = adapter
        self.scene = scene

    def __getattr__(self, name):
        boundary = get_boundary(self.entity, name)
        return partial(self._call, name, boundary)

    def _call(self, name, boundary, *args, **kwargs):
        guard = self.scene.guard
        if self.scene.is_observing:
            if boundary.value_parameter is not None:
                guard.check((args, kwargs))
                return getattr(self.adapter, name)(*args, **kwargs)
            guard.is_reading = True
            try:
                return getattr(self.adapter, name)(*args, **kwargs).clone()
            finally:
                guard.is_reading = False
        if boundary.value_parameter is None:
            value = getattr(self.entity, name)(*args, **kwargs)
            guard.track(value)
            return value
        return None


def _skip(*args, **kwargs):
    return None


def _snapshot(value, memo):
    """Own tensor/container storage while preserving repeated references in an environment snapshot."""
    pending = [value]
    visited = set()
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in memo or identity in visited:
            continue
        visited.add(identity)
        if isinstance(item, torch.Tensor):
            memo[identity] = item.clone()
        elif isinstance(item, np.ndarray):
            memo[identity] = item.copy()
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, MethodType) and id(item.__self__) in memo:
            memo[identity] = MethodType(item.__func__, memo[id(item.__self__)])
        elif type(item).__module__.startswith("genesis.") and type(item).__module__ != __name__:
            memo[identity] = _UnavailableObject(type(item).__name__)
        elif type(item).__module__ == __name__ or isinstance(
            item,
            (
                type(None),
                str,
                bytes,
                bool,
                int,
                float,
                complex,
                slice,
                range,
                type,
                Path,
                Enum,
                torch.device,
                torch.dtype,
                torch.layout,
                torch.memory_format,
                np.generic,
                np.dtype,
                FunctionType,
                BuiltinFunctionType,
            ),
        ):
            memo[identity] = item
        else:
            raise TypeError(
                f"Unsupported environment attribute type '{type(item).__name__}'. Store tensors and ordinary "
                "containers directly on the environment; custom mutable resource wrappers cannot be captured."
            )
    return copy.deepcopy(value, memo)


def _validate_snapshot(env):
    pending = list(vars(env).items())
    visited = set()
    storage_owners = {}
    while pending:
        path, value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if isinstance(value, torch.Tensor) and value.layout == torch.strided:
            storage = value.untyped_storage()
            if storage.nbytes() == 0:
                continue
            key = (value.device, storage.data_ptr())
            if key in storage_owners:
                raise ValueError(
                    f"Environment tensors '{storage_owners[key]}' and '{path}' share storage. Store one tensor and "
                    "create its views inside the annotated inference methods instead of caching distinct views."
                )
            storage_owners[key] = path
        elif isinstance(value, dict):
            pending.extend((f"{path}[{key!r}]", item) for key, item in value.items())
        elif isinstance(value, (list, tuple)):
            pending.extend((f"{path}[{index}]", item) for index, item in enumerate(value))


def export_environment(
    env,
    policy,
    output_dir: str | Path,
    *,
    sample_inputs: Iterable[Mapping[str, torch.Tensor]] | None = None,
    capture_steps: int = 5,
    name: str = "policy",
    initial_state: Mapping[str, torch.Tensor] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Path:
    """Export the original environment step and policy as one ONNX model.

    ``env.get_observations`` must carry ``@policy_environment``. Capture runs ``env.step`` on an owned snapshot,
    discarding its initial actuator writes and intercepting its single ``scene.step`` call. At that boundary, declared
    external inputs and feedback are injected; original observation and history computations then execute normally.
    The policy consumes the returned observations. A second original ``step`` captures action processing and stops at
    the physics boundary. No capture pass advances physics or writes to live entities.

    ``skip`` methods are disabled only on the owned snapshot to exclude command sampling and episode resets. Tensor
    attributes and ordinary containers are copied. Custom mutable objects are unsupported; unbound Genesis objects
    cannot be accessed. Pre-physics tensor locals cannot enter post-physics observations: the exporter rejects this
    dependency instead of freezing discarded values. Repeated references to one tensor are supported, but distinct cached views sharing storage are
    rejected. Views created inside ``step`` remain traced. A native step must call ``scene.step`` exactly once and
    return observations, either directly or as the first tuple item.

    Initialize the environment before export. Declared state attributes supply deployment reset values unless
    ``initial_state`` is given; these values represent the boundary immediately after physics, before observations and
    history updates. The policy callable must include normalization and expose any persistent tensor state explicitly.

    By default, collect ``capture_steps`` samples, advancing the original environment with its ordinary policy/step
    between samples. This changes its simulation state. An explicit ``sample_inputs`` iterable controls collection
    instead; its mappings must contain exactly the declared external names and may advance the scene between yields.
    """
    metadata = get_policy_environment(env)
    _validate_snapshot(env)
    entities = {attribute: getattr(env, attribute) for attribute in metadata.entities}
    inputs = {item.name: item for item in metadata.inputs}
    if len(inputs) != len(metadata.inputs) or len({item.attribute for item in metadata.inputs}) != len(metadata.inputs):
        raise ValueError("Environment external input names and attributes must be unique.")
    attributes = (*metadata.entities, *(item.attribute for item in metadata.inputs), *metadata.state)
    if len(set(attributes)) != len(attributes):
        raise ValueError("Environment entity, external input and feedback attributes must be distinct.")
    if initial_state is None:
        initial_state = {attribute: getattr(env, attribute) for attribute in metadata.state}
    if set(initial_state) != set(metadata.state):
        raise ValueError("Initial state must contain exactly the annotated environment feedback attributes.")
    if any(not isinstance(value, torch.Tensor) for value in initial_state.values()):
        raise TypeError("Environment feedback attributes must be Torch tensors.")

    def samples():
        if sample_inputs is not None:
            for sample in sample_inputs:
                if set(sample) != set(inputs):
                    raise ValueError("Samples must contain exactly the annotated environment external input names.")
                yield _snapshot(sample, {})
            return
        if not isinstance(capture_steps, int) or capture_steps < 1:
            raise ValueError("capture_steps must be a positive integer.")
        for i_step in range(capture_steps):
            if i_step:
                env.step(policy(env.get_observations()))
            sample = {name: getattr(env, item.attribute) for name, item in inputs.items()}
            if any(not isinstance(value, torch.Tensor) for value in sample.values()):
                raise TypeError("Environment external input attributes must be Torch tensors.")
            yield _snapshot(sample, {})

    def step(adapters, external, state):
        _validate_snapshot(env)
        owned = copy.copy(env)
        attributes_before_step = {}

        def begin_observation():
            guard.is_injecting = True
            try:
                injected = _snapshot(state, {})
                injected.update((inputs[name].attribute, value) for name, value in _snapshot(external, {}).items())
                memo = {id(owned): owned}
                for attribute, value in injected.items():
                    memo[id(attributes_before_step[attribute])] = value
                    memo[id(getattr(owned, attribute))] = value
                owned.__dict__ = _snapshot(vars(owned), memo)
            finally:
                guard.is_injecting = False

        scene = next(iter(entities.values())).scene
        capture_scene = _CaptureScene(scene, begin_observation)
        guard = capture_scene.guard = _PreambleGuard(capture_scene)
        memo = {id(env): owned, id(scene): capture_scene}
        memo.update((id(entity), _UnavailableObject(type(entity).__name__)) for entity in scene.entities)
        memo.update(
            (id(entities[name]), _CaptureEntity(entities[name], adapter, capture_scene))
            for name, adapter in adapters.items()
        )
        owned.__dict__ = _snapshot(vars(env), memo)
        attributes_before_step.update(
            (attribute, getattr(owned, attribute))
            for attribute in (*state, *(item.attribute for item in inputs.values()))
        )
        for method_name in metadata.skip:
            setattr(owned, method_name, _skip)
        action_template = getattr(owned, metadata.action)
        if not isinstance(action_template, torch.Tensor):
            raise TypeError("The annotated environment action attribute must be a Torch tensor.")
        dummy_action = torch.zeros_like(action_template)
        guard.track(vars(owned), is_derived=False)
        guard.track(dummy_action)
        with guard:
            try:
                result = owned.step(dummy_action)
            except _CaptureComplete:
                raise ValueError("Native environment step must call scene.step exactly once.") from None
            if not capture_scene.is_observing:
                raise ValueError("Native environment step must call scene.step exactly once.")
            observations = result[0] if isinstance(result, tuple) else result
            if observations is None:
                raise ValueError("Native environment step must return policy observations.")
            guard.check(observations)
            actions = policy(observations)
            try:
                owned.step(actions)
            except _CaptureComplete:
                state_next = {attribute: getattr(owned, attribute) for attribute in metadata.state}
                guard.check(state_next)
                return state_next
            raise ValueError("Native action processing did not reach the scene.step boundary.")

    return export_policy(
        step,
        entities,
        samples(),
        initial_state,
        output_dir,
        step_dt=getattr(env, metadata.step_dt),
        name=name,
        input_semantics={name: item.semantics for name, item in inputs.items()},
        rtol=rtol,
        atol=atol,
    )
