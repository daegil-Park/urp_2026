# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for UR10e with a spindle tool (local USD scene).

- Loads the local USD: /home/eunseop/isaac/isaac_save/ur10e_concave_surface.usd
- Spawns only the articulation prim: /World/ur10e_w_spindle_robot
- EE (tool) frame: wrist_3_link
- Two variants: default PD and HIGH_PD (task-space / diff-IK friendly)

Notes
-----
* If your USD contains a full stage (PhysicsScene, lights, etc.), setting
  `prim_path` ensures only the robot prim is imported as the articulation.
* If joint names in the USD differ, update `UR10E_HOME_DICT` and
  `UR10E_ARM_JOINTS` accordingly.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# -----------------------------------------------------------------------------
# User paths and prims
# -----------------------------------------------------------------------------

UR10E_USD_PATH = "/home/eunseop/isaac/isaac_save/ur10e_tuning2.usd"
# UR10E_PRIM_PATH = "/World/Robot"   # as in your USD
EE_FRAME_NAME = "wrist_3_link"                      # end-effector frame in your USD

# -----------------------------------------------------------------------------
# Home pose (rad)
# -----------------------------------------------------------------------------
UR10E_HOME_DICT = {
    "shoulder_pan_joint": 0,
    "shoulder_lift_joint": -1.57079632679,
    "elbow_joint": -1.57079632679,
    "wrist_1_joint": -1.57079632679,
    "wrist_2_joint": 1.57079632679,
    "wrist_3_joint": 0.0,

    # "shoulder_pan_joint": 0.0,
    # "shoulder_lift_joint": 0.0,
    # "elbow_joint": 0.0,
    # "wrist_1_joint": 0.0,
    # "wrist_2_joint": 0.0,
    # "wrist_3_joint": 0.0,
}

# Actuator joint list (exact names expected in the USD)
UR10E_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# -----------------------------------------------------------------------------
# Base configuration
# -----------------------------------------------------------------------------
UR10E_W_SPINDLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR10E_USD_PATH,
        # prim_path=UR10E_PRIM_PATH,  # import only this prim as an articulation
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # If collisions feel too "thick", uncomment and tune:
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.005, rest_offset=0.0
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=UR10E_HOME_DICT,
        # Optionally set base pose (if USD doesn't already place it):
        # pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)
    ),
    actuators={
        "ur10e_arm": ImplicitActuatorCfg(
            joint_names_expr=UR10E_ARM_JOINTS,
            # Conservative effort; adjust if your USD defines different limits
            effort_limit_sim=150.0,
            stiffness=120.0,
            damping=8.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Expose EE frame name for downstream controllers / tasks
UR10E_W_SPINDLE_CFG.ee_frame_name = EE_FRAME_NAME

# -----------------------------------------------------------------------------
# High-PD variant (helpful for task-space/diff-IK style control)
# -----------------------------------------------------------------------------
UR10E_W_SPINDLE_HIGH_PD_CFG = UR10E_W_SPINDLE_CFG.copy()
UR10E_W_SPINDLE_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR10E_W_SPINDLE_HIGH_PD_CFG.actuators["ur10e_arm"].stiffness = 400.0
UR10E_W_SPINDLE_HIGH_PD_CFG.actuators["ur10e_arm"].damping = 60.0
UR10E_W_SPINDLE_HIGH_PD_CFG.ee_frame_name = EE_FRAME_NAME

# Optional explicit exports
__all__ = [
    "UR10E_W_SPINDLE_CFG",
    "UR10E_W_SPINDLE_HIGH_PD_CFG",
    "EE_FRAME_NAME",
]

