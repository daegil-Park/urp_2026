# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import copy
import os

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

# Custom MDP modules
import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew      

from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# -------------------------------------------------------------------------
# [수직 자세] 기본 자세 정의
# -------------------------------------------------------------------------
DEVICE_READY_STATE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708, 
    "elbow_joint": 1.5708,          
    "wrist_1_joint": -1.5708,       
    "wrist_2_joint": -1.5708,       
    "wrist_3_joint": 0.0,
}

# -------------------------------------------------------------------------
# Scene Configuration
# -------------------------------------------------------------------------

TEMP_ROBOT_CFG = copy.deepcopy(UR10E_W_SPINDLE_CFG)

# 로봇 자체의 충돌 센서 활성화
TEMP_ROBOT_CFG.spawn = sim_utils.UsdFileCfg(
    usd_path=UR10E_W_SPINDLE_CFG.spawn.usd_path,
    activate_contact_sensors=True, 
    rigid_props=UR10E_W_SPINDLE_CFG.spawn.rigid_props,
    articulation_props=UR10E_W_SPINDLE_CFG.spawn.articulation_props,
)

@configclass
class RobotarmSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # 1. Ground (바닥)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # 2. Workpiece (물리적 실체가 있는 상자로 교체)
    # [중요] 기존 USD 대신 CuboidCfg를 사용하여 물리 충돌을 강제합니다.
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.1),  # 가로 0.5m, 세로 0.5m, 높이 0.1m
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)), # 회색
            
            # [물리 속성 핵심]
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, # True = 고정된 물체 (로봇이 밀어도 안 밀림)
                disable_gravity=True,
            ),
            # [충돌 속성 핵심] 이 줄이 없으면 유령입니다.
            collision_props=sim_utils.CollisionPropertiesCfg(), 
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # 상자의 중심 좌표. 높이 0.1m 상자이므로 Z=0.05에 두면 바닥 위에 딱 붙습니다.
            pos=(0.75, 0.0, 0.05), 
        ),
    )

    # 3. Robot
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            joint_pos=DEVICE_READY_STATE, 
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=200.0,   # 강성을 높여서 흐물거림 방지
                damping=40.0,     
            ),
        }
    )

    # 4. Sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*", 
        history_length=3,
        track_air_time=False,
    )
    
    # 5. Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# -------------------------------------------------------------------------
# MDP Settings
# -------------------------------------------------------------------------

@configclass
class CommandsCfg:
    pass

@configclass
class ActionsCfg:
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.02, # 액션 스케일
        use_default=True, 
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        path_tracking = ObsTerm(func=local_obs.path_tracking_obs)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_history = ObsTerm(func=local_obs.ee_pose_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # Reward Weights
    track_path = RewTerm(func=local_rew.track_path_reward, weight=10.0, params={"sigma": 0.1})
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=20.0)
    joint_reg = RewTerm(func=local_rew.joint_deviation_reward, weight=2.0)
    force_control = RewTerm(func=local_rew.force_control_reward, weight=1.0, params={"target_force": 10.0})

    # Penalties
    joint_vel = RewTerm(func=local_rew.joint_vel_penalty, weight=-0.01)
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.01)
    
    # 영역 이탈 페널티
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-5.0)


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=local_rew.reset_robot_to_cobra, 
        mode="reset",
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # [추가] 물리적 한계선 (Safety Net)
    # 로봇의 몸체(Root)나 관절이 바닥(Z=0) 아래로 내려가면 에피소드 종료
    # 이는 물리 엔진이 뚫리더라도 학습 로직에서 잡아냅니다.
    underground_death = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.02, "asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=64, env_spacing=2.5) # 디버깅을 위해 개수 줄임
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4            
        self.episode_length_s = 15.0   
        self.sim.dt = 1.0 / 120.0 
        self.sim.render_interval = self.decimation
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
