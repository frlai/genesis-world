"""Scene construction shared by walking training and direct deployment."""

import genesis as gs


def build_scene(env_cfg, *, num_envs, dt, show_viewer=False):
    """Build the Go2 training scene and return its scene and robot."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
            # can save memory. Violating this condition will raise an exception.
            max_collision_pairs=20,
            tolerance=1e-5,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    # add plain
    scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
        )
    )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=env_cfg["base_init_pos"],
            quat=env_cfg["base_init_quat"],
        ),
    )

    scene.build(n_envs=num_envs)

    return scene, robot
