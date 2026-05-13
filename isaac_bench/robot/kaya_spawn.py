from __future__ import annotations


def spawn_kaya(world, prim_path, name, usd_path, position, orientation):
    """Spawn Kaya inside Isaac Sim. Imports are intentionally local."""
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot

    return world.scene.add(
        WheeledRobot(
            prim_path=prim_path,
            name=name,
            wheel_dof_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            create_robot=True,
            usd_path=usd_path,
            position=position,
            orientation=orientation,
        )
    )

