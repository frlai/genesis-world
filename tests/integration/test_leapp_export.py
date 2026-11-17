import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat
from genesis.utils.leapp import TensorMetadata, tensor_input
from genesis.utils.leapp.types import dof_names, load_metadata

from ..utils.assertions import assert_allclose, assert_equal

leapp = pytest.importorskip("leapp")

from genesis.utils.leapp.export import export_policy


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("external_command", [pytest.param(False, marks=pytest.mark.precision("32")), True])
def test_complete_policy_export(tmp_path, n_envs, external_command, record_property):
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            fixed=True,
        ),
    )
    scene.build(n_envs=n_envs)
    robot.set_dofs_kp([10.0, 20.0])
    robot.set_dofs_kv([0.5, 1.0])
    shape = (n_envs, 2) if n_envs else (2,)
    shape_command = (*shape[:-1], 3)
    shape_history = (*shape[:-1], 2, 3)
    actor = torch.nn.Sequential(
        torch.nn.Linear(10, 8, device=gs.device, dtype=gs.tc_float),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2, device=gs.device, dtype=gs.tc_float),
    ).eval()
    observation_mean = torch.linspace(-0.3, 0.3, 10, device=gs.device, dtype=gs.tc_float)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float)
    command_constant = torch.zeros(shape_command, device=gs.device, dtype=gs.tc_float)

    def compute(position, velocity, orientation, command, previous):
        projected_gravity = transform_by_quat(gravity.expand_as(command), inv_quat(orientation))
        observation = torch.cat((position * 0.5, velocity * 0.1, projected_gravity, command), dim=-1)
        action = torch.tanh(actor((observation - observation_mean) * 0.8) + previous * 0.25)
        target = torch.clip(action, -0.7, 0.7) * 0.3 + 0.1
        return action, target

    def step(entities, inputs, state):
        entity = entities["arm"]
        action, target = compute(
            entity.get_dofs_position([1, 0]),
            entity.get_dofs_velocity([1, 0]),
            entity.get_quat(relative=False),
            inputs["command"].mean(dim=-2) if external_command else command_constant,
            state["previous"],
        )
        entity.control_dofs_position(target, [1, 0])
        return {"previous": action}

    def samples():
        for value in (0.1, -0.2, 0.3):
            robot.set_dofs_position([value, -value])
            robot.set_dofs_velocity([value * 0.3, value * 0.7])
            yield (
                {"command": torch.full(shape_history, value, device=gs.device, dtype=gs.tc_float)}
                if external_command
                else {}
            )

    inputs_metadata = {
        "command": TensorMetadata(
            kind="frame/twist", component_names=("x", "y", "angular_z"), is_setpoint=True, expressed_in="world"
        )
    }
    bundle = export_policy(
        step,
        {"arm": robot},
        samples(),
        {"previous": torch.zeros(shape, device=gs.device, dtype=gs.tc_float)},
        tmp_path,
        step_dt=scene.dt,
        input_semantics=inputs_metadata,
        rtol=2e-6,
        atol=2e-7,
    )
    metadata = load_metadata(bundle)
    assert metadata.controllers[0].dof_names == tuple(reversed(dof_names(robot)))
    assert_equal(metadata.controllers[0].kp, [20.0, 10.0])
    assert_equal(metadata.controllers[0].kv, [1.0, 0.5])
    manager = leapp.InferenceManager(str(bundle))
    assert_equal(len(manager.nodes), 1)
    descriptions = {description["name"]: description for description in manager.nodes["policy"].input_descriptions}
    assert descriptions["arm_get_quat_0"]["axes"][-1]["names"] == ["qw", "qx", "qy", "qz"]
    assert descriptions["arm_get_quat_0"]["genesis_connection"]["arguments"] == [{"name": "relative", "value": False}]
    assert descriptions["arm_get_dofs_position_0"]["axes"][-1]["names"] == list(reversed(dof_names(robot)))
    if external_command:
        assert descriptions["external_command"]["is_setpoint"]
        assert descriptions["external_command"]["axes"][-2] is None
        assert descriptions["external_command"]["axes"][-1]["names"] == ["x", "y", "angular_z"]
    generator = torch.Generator(device=gs.device).manual_seed(17)
    error_target_max = 0.0
    error_state_max = 0.0
    for _ in range(2):
        manager.reset()
        previous = torch.zeros(shape, device=gs.device, dtype=gs.tc_float)
        for _ in range(8):
            position = torch.rand(shape, generator=generator, device=gs.device, dtype=gs.tc_float) - 0.5
            velocity = torch.rand(shape, generator=generator, device=gs.device, dtype=gs.tc_float) * 2 - 1
            orientation = torch.rand((*shape[:-1], 4), generator=generator, device=gs.device, dtype=gs.tc_float) - 0.5
            orientation /= torch.linalg.vector_norm(orientation, dim=-1, keepdim=True)
            command = (
                torch.rand(shape_history, generator=generator, device=gs.device, dtype=gs.tc_float) - 0.5
                if external_command
                else command_constant
            )
            inputs = {
                "policy/arm_get_dofs_position_0": position,
                "policy/arm_get_dofs_velocity_0": velocity,
                "policy/arm_get_quat_0": orientation,
            }
            if external_command:
                inputs["policy/external_command"] = command
            with torch.no_grad():
                previous, target = compute(
                    position, velocity, orientation, command.mean(dim=-2) if external_command else command, previous
                )
            outputs = manager.run_policy(inputs)
            target_actual = outputs["policy/arm_control_dofs_position_0"]
            state_actual = manager.value_dict["policy"]["state_previous_in"]
            error_target_max = max(
                error_target_max, (target_actual - target.to(target_actual.device)).abs().max().item()
            )
            error_state_max = max(error_state_max, (state_actual - previous.to(state_actual.device)).abs().max().item())
            assert_allclose(target_actual, target, tol=2e-7)
            assert_allclose(state_actual, previous, tol=5e-7)
    record_property("error_target_max", error_target_max)
    record_property("error_state_max", error_state_max)


@pytest.mark.required
def test_annotation_preserves_execution():
    def getter(value):
        return value + 2

    annotated = tensor_input(kind="joint/position")(getter)
    assert annotated is getter
    assert_equal(annotated(3), 5)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(gs.__file__.rsplit("/genesis/", 1)[0])
    process = subprocess.run(
        [sys.executable, "-c", "import genesis, sys; assert 'leapp' not in sys.modules"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert_equal(process.returncode, 0)


@pytest.mark.required
def test_capture_failure_restores_execution(tmp_path):
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            fixed=True,
        ),
    )
    scene.build(n_envs=1)
    position = robot.get_dofs_position().clone()

    def invalid_step(entities, inputs, state):
        entities["arm"].get_dofs_position()
        raise ValueError("Invalid application input.")

    with pytest.raises(ValueError, match="Invalid application input"):
        export_policy(invalid_step, {"arm": robot}, [{}], {}, tmp_path, step_dt=scene.dt)
    assert_equal(robot.get_dofs_position(), position)
    assert_equal(torch.ones(2) + 1, np.array([2.0, 2.0]))

    def valid_step(entities, inputs, state):
        entities["arm"].control_dofs_force(entities["arm"].get_dofs_position() * 0.1)
        return {}

    bundle = export_policy(valid_step, {"arm": robot}, [{}, {}], {}, tmp_path, step_dt=scene.dt)
    manager = leapp.InferenceManager(str(bundle))
    inputs = manager.get_mock_input()
    inputs["policy/arm_get_dofs_position_0"][:] = 0.5
    outputs = manager.run_policy(inputs)
    assert_allclose(outputs["policy/arm_control_dofs_force_0"], np.full((1, 2), 0.05), atol=5e-9)
