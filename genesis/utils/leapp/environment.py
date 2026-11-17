"""Capture annotated environment methods without duplicating their inference computations."""

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from .annotations import get_policy_environment
from .export import export_policy


class _CaptureScene:
    def __getattr__(self, name):
        raise ValueError("Environment inference stages cannot access the scene. Keep physics outside capture.")


def _snapshot(value, memo):
    """Own tensor/container storage while preserving repeated references in an environment snapshot."""
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone()
    elif isinstance(value, np.ndarray):
        result = value.copy()
    elif isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((key, _snapshot(item, memo)) for key, item in value.items())
    elif isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(_snapshot(item, memo) for item in value)
    elif isinstance(value, tuple):
        result = tuple(_snapshot(item, memo) for item in value)
    elif isinstance(value, MethodType) and id(value.__self__) in memo:
        result = MethodType(value.__func__, memo[id(value.__self__)])
    else:
        return value
    memo[id(value)] = result
    return result


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
    """Export the original annotated observation, inference and actuator methods as one ONNX model.

    ``env.get_observations`` must carry ``@policy_environment``. Its ``prepare`` methods run in declared order,
    followed by ``policy(env.get_observations())`` and the declared action method. Capture uses fresh owned copies of
    tensor attributes and ordinary tensor containers, replacing entity attributes with export adapters. Other custom
    objects are shared read-only; inference methods must not mutate them or invoke physics, resets or command sampling.

    Initialize the environment before exporting: its declared state attributes provide deployment reset values unless
    ``initial_state`` is supplied. State updates, including in-place history writes, remain inside the original methods.
    The policy callable must include inference normalization and expose any persistent tensor state explicitly.

    By default, collect ``capture_steps`` samples, advancing the original environment with its ordinary policy/step
    between samples. This changes its simulation state. An explicit ``sample_inputs`` iterable instead controls sample
    collection; its mappings must contain exactly the declared external names and may advance the scene between yields.
    """
    metadata = get_policy_environment(env)
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
            yield {name: getattr(env, item.attribute).clone() for name, item in inputs.items()}

    def step(adapters, external, state):
        owned = copy.copy(env)
        scene = next(iter(entities.values())).scene
        memo = {id(env): owned, id(scene): _CaptureScene()}
        memo.update((id(entities[name]), adapter) for name, adapter in adapters.items())
        injected = _snapshot(state, {})
        injected.update((inputs[name].attribute, value) for name, value in _snapshot(external, {}).items())
        memo.update((id(getattr(env, attribute)), value) for attribute, value in injected.items())
        owned.__dict__ = {attribute: _snapshot(value, memo) for attribute, value in vars(env).items()}
        for method_name in metadata.prepare:
            getattr(owned, method_name)()
        actions = policy(owned.get_observations())
        getattr(owned, metadata.action)(actions)
        return {attribute: getattr(owned, attribute) for attribute in metadata.state}

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
