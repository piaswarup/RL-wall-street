# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configuration for the Vertigo-style wall-climbing robot.

The USD is generated from ``scripts/make_vertigo_urdf.py`` -> ``vertigo.urdf`` -> ``usd/vertigo/vertigo.usda``.
Re-run both scripts after changing the robot geometry.
"""

from pathlib import Path

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

VERTIGO_USD_PATH = Path(__file__).resolve().parent / "data" / "vertigo" / "usd" / "vertigo" / "vertigo.usda"

TILT_JOINTS = ["front_tilt", "rear_tilt"]
WHEEL_JOINTS = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
PROP_BODIES = ["front_prop", "rear_prop"]

VERTIGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(VERTIGO_USD_PATH),
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            PhysxRigidBodyCfg(
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
        ],
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.07), joint_pos={".*": 0.0}),
    actuators={
        # position-controlled 360-degree tilt servos (one per propeller, axis = body Y)
        "tilt_servos": ImplicitActuatorCfg(
            joint_names_expr=TILT_JOINTS,
            joint_effort_limit=5.0,
            joint_velocity_limit=8.0,
            stiffness=20.0,
            damping=1.0,
        ),
        # free-rolling wheels: no motor, just a little bearing friction
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,
            joint_effort_limit=0.01,
            stiffness=0.0,
            damping=0.002,
        ),
    },
)
"""Vertigo-style robot: 4 free-rolling wheels + 2 propellers on 360-degree tilt joints (axis = body Y)."""
