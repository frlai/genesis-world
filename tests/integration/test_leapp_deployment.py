from dataclasses import replace

import pytest
import torch

import genesis as gs
from genesis.utils.leapp.types import load_metadata, save_metadata

from ..utils.assertions import assert_allclose, assert_equal

pytest.importorskip("leapp", minversion="0.8.1")

from genesis.utils.leapp.deployment import LeappDeployment
from genesis.utils.leapp.export import export_policy


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("n_envs", [0, 2])
def test_policy_control_and_reset(n_envs, show_viewer, tmp_path):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 2.0),
            camera_lookat=(1.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(2.0, 0.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        vis_mode="collision",
    )
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            pos=(0.0, 0.0, 1.0),
            file="urdf/drones/cf2x.urdf",
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=n_envs)
    shape = (n_envs, 2) if n_envs else (2,)
    command = torch.full(shape, fill_value=0.4, dtype=gs.tc_float, device=gs.device)
    rpm = torch.full((*shape[:-1], 4), fill_value=15000.0, dtype=gs.tc_float, device=gs.device)
    initial_state = {"memory": torch.zeros_like(command)}
    initial_command = command.clone()
    initial_rpm = rpm.clone()

    def policy_step(entities, inputs, state):
        position = entities["cube"].get_pos()
        force = inputs["command"] + state["memory"] - 0.2 * position[..., :2]
        entities["cube"].control_dofs_force(force, dofs_idx_local=[1, 0])
        entities["drone"].set_propellers_rpm(inputs["rpm"] * (1.0 + inputs["command"][..., :1] * 0.01))
        return {"memory": 0.5 * state["memory"] + 0.1 * inputs["command"]}

    entities = {"cube": cube, "drone": drone}
    samples = [{"command": command * scale, "rpm": rpm} for scale in (0.5, 1.0, 1.5)]
    bundle = export_policy(policy_step, entities, samples, initial_state, tmp_path, step_dt=2 * scene.dt)
    deployment = LeappDeployment(scene, bundle, entities)
    deployment.reset()
    memory = initial_state["memory"].clone()
    initial_position = cube.get_pos().clone()
    drone_mass = drone.get_mass_mat()[..., 2, 2]
    drone_damping = drone.get_dofs_damping(dofs_idx_local=2)[..., 0]
    drone_weight = drone.get_mass() * -9.81
    drone_velocity = torch.zeros_like(drone_weight)
    first_force = None
    for i_step in range(4):
        command_step = command * (1 + 0.2 * i_step)
        expected_force = command_step + memory - 0.2 * cube.get_pos()[..., :2]
        expected_rpm = rpm * (1.0 + command_step[..., :1] * 0.01)
        thrust = drone.KF * torch.square(expected_rpm).sum(dim=-1)
        for _ in range(2):
            drone_velocity = (drone_mass * drone_velocity + scene.dt * (thrust + drone_weight)) / (
                drone_mass + drone_damping * scene.dt
            )
        deployment.step({"command": command_step, "rpm": rpm})
        assert_allclose(cube.get_dofs_control_force(dofs_idx_local=[1, 0]), expected_force, tol=2e-6)
        assert_allclose(drone.get_dofs_velocity(dofs_idx_local=2)[..., 0], drone_velocity, tol=2e-6)
        if i_step == 0:
            first_force = expected_force.clone()
        memory = 0.5 * memory + 0.1 * command_step
    assert (cube.get_dofs_velocity(dofs_idx_local=0) > 0).all()
    assert_allclose(scene.get_time(), 8 * scene.dt, atol=1e-8)

    deployment.reset()
    assert_equal(command, initial_command)
    assert_equal(rpm, initial_rpm)
    assert_equal(cube.get_pos(), initial_position)
    deployment.step({"command": command, "rpm": rpm})
    assert_allclose(cube.get_dofs_control_force(dofs_idx_local=[1, 0]), first_force, tol=2e-6)

    with pytest.raises(ValueError, match="External input names"):
        deployment.read_inputs({"command": command})
    with pytest.raises(ValueError, match="expects shape"):
        deployment.read_inputs({"command": command[..., :1], "rpm": rpm})
    with pytest.raises(ValueError, match="requires entity"):
        LeappDeployment(scene, bundle, {"cube": cube})
    deployment.close()
    with pytest.raises(RuntimeError, match="closed"):
        deployment.step({"command": command, "rpm": rpm})
    assert scene.is_built


@pytest.mark.required
@pytest.mark.precision("32")
def test_controller_configuration(show_viewer, tmp_path):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 1.5),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=1)
    cube.set_dofs_kp([20.0, 10.0], dofs_idx_local=[1, 0])
    cube.set_dofs_kv([2.0, 1.0], dofs_idx_local=[1, 0])
    cube.set_dofs_force_range([-5.0, float("-inf")], [5.0, float("inf")], dofs_idx_local=[1, 0])
    command = torch.tensor([[0.1, 0.2]], dtype=gs.tc_float, device=gs.device)

    def policy_step(entities, inputs, state):
        positions = entities["cube"].get_dofs_position(dofs_idx_local=[1, 0])
        entities["cube"].control_dofs_position(positions + inputs["command"], dofs_idx_local=[1, 0])
        return state

    entities = {"cube": cube}
    bundle = export_policy(policy_step, entities, [{"command": command}], {}, tmp_path, step_dt=scene.dt)
    cube.set_dofs_kp(0.0)
    cube.set_dofs_kv(0.0)
    deployment = LeappDeployment(scene, bundle, entities)
    assert_equal(cube.get_dofs_kp(dofs_idx_local=[1, 0]), [20.0, 10.0])
    assert_equal(cube.get_dofs_kv(dofs_idx_local=[1, 0]), [2.0, 1.0])
    force_lower, force_upper = cube.get_dofs_force_range(dofs_idx_local=[1, 0])
    assert_equal(force_lower, [-5.0, float("-inf")])
    assert_equal(force_upper, [5.0, float("inf")])
    deployment.step({"command": command})
    assert (cube.get_dofs_velocity(dofs_idx_local=[1, 0]) > 0).all()

    metadata = load_metadata(bundle)
    save_metadata(bundle, replace(metadata, step_dt=1.5 * scene.dt))
    with pytest.raises(ValueError, match="integer multiple"):
        LeappDeployment(scene, bundle, entities)
    save_metadata(bundle, replace(metadata, physics_dt=2 * scene.dt))
    with pytest.raises(ValueError, match="differs from exported"):
        LeappDeployment(scene, bundle, entities)
