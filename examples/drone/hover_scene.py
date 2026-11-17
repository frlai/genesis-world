"""Scene construction shared by hovering training and direct deployment."""

import genesis as gs


def build_scene(env_cfg, *, num_envs, dt, show_viewer=False):
    """Build the hovering scene and return its scene, drone, target marker, and camera."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt / 2,
            enable_collision=True,
            enable_joint_limit=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=list(range(min(10, num_envs))),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    if env_cfg["visualize_target"]:
        target = scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.05,
                fixed=False,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.5, 0.5),
                ),
            ),
        )
    else:
        target = None

    cam = None
    if env_cfg["visualize_camera"]:
        cam = scene.add_camera(
            res=(640, 480),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=True,
        )

    # add drone
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
        )
    )

    scene.build(n_envs=num_envs)

    return scene, drone, target, cam
