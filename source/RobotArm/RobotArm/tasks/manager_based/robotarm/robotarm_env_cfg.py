# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from . import mdp

# reward, observation modul import
import importlib
local_obs = importlib.import_module("RobotArm.tasks.manager_based.robotarm.mdp.observations")
local_rew = importlib.import_module("RobotArm.tasks.manager_based.robotarm.mdp.rewards")

##
# Pre-defined configs
##

from RobotArm.robots.ur10e_w_spindle import *

# solver = nrs_ik_core.IKSolver(tool_z=0.2, use_degrees=True)
# angle = solver.compute(pose)

##
# Scene definition
##


@configclass
class RobotarmSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/eunseop/isaac/isaac_save/flat_surface_2.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # robot
    robot: ArticulationCfg = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pass
    # ee_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=EE_FRAME_NAME,
    #     resampling_time_range=(0.5, 1.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.0, 0.5),   # 작업 면적
    #         pos_y=(0.0, 1.0),
    #         pos_z=(0.05, 0.1),
    #         roll=(-3.14, 3.14),
    #         pitch=(-1.57, 1.57),  # depends on end-effector axis
    #         yaw=(-3.14, 3.14),
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    #joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        #use_default=True,
        scale=0.5,
    )
    gripper_action: ActionTerm | None = None

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        
        grid_mask_state = ObsTerm(      # Grid Mask의 상태: Policy가 방문하지 않은 곳을 찾아가도록 유도
            func=local_obs.grid_mask_state_obs,
            params={
                # "grid_mask": SceneEntityCfg("env", data_key="grid_mask"), 
                    # "grid_mask": SceneEntityCfg(name="workpiece", body_names=["grid_mask"]),
                "grid_mask_history_len": 4
            }
        )

        ee_pose_history = ObsTerm(
            func=local_obs.ee_pose_history,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[EE_FRAME_NAME]),
                "history_len": 50
                },
        )

        # contact_forces = ObsTerm(
        #     func=local_obs.get_contact_forces,
        #     params={
        #         "sensor_name": "ActionGraphContactSensor"
        #     }
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_grid_mask = EventTerm(
        func=mdp.reset_grid_mask,  # 1번에서 정의한 함수
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # 커버리지 증가 보상 및 중복 방문 패널티
    coverage = RewTerm(
        func=local_rew.coverage_reward,
        weight=2.0,
        params={"grid_size": 0.02},
    )
    revisit_penalty = RewTerm(
        func=local_rew.revisit_penalty,
        weight=-0.5,
        params={"grid_size": 0.02},
    )

    # 안정성 (EE의 z 높이 및 자세)
    surface_proximity = RewTerm(
        func=local_rew.surface_proximity_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=EE_FRAME_NAME)},
    )
    ee_orientation_alignment = RewTerm(
        func=local_rew.ee_orientation_alignment,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_FRAME_NAME]),
            "target_axis": (0.0, 0.0, -1.0), # EE의 Z축이 월드 Z축과 평행하도록 (수직 폴리싱 가정)
        },
    )

    # 커버리지 100% 달성 시 큰 보상
    coverage_completion = RewTerm(
        func=local_rew.coverage_completion_reward,
        weight=1.0,
        params={"threshold": 0.90, "bonus_scale": 10.0}
    )

    # 효율성
    time_efficiency = RewTerm(
        func=local_rew.time_efficiency_reward,
        weight=1.0,
        params={"max_steps": 750},  # episode 길이에 맞춰 조정
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # # 작업 성공
    # task_success = DoneTerm(
    #     func=local_rew.coverage_completion_reward,
    #     params={
    #         "threshold": 0.95,
    #         "bonus_scale": 10.0,
    #     },
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    # )



##
# Environment configuration
##


@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=128, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
