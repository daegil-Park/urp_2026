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

import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew      

from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# -------------------------------------------------------------------------
# Scene Configuration
# -------------------------------------------------------------------------

TEMP_ROBOT_CFG = copy.deepcopy(UR10E_W_SPINDLE_CFG)

# Contact Sensor 설정
TEMP_ROBOT_CFG.spawn = sim_utils.UsdFileCfg(
    usd_path=UR10E_W_SPINDLE_CFG.spawn.usd_path,
    activate_contact_sensors=True, 
    rigid_props=UR10E_W_SPINDLE_CFG.spawn.rigid_props,
    articulation_props=UR10E_W_SPINDLE_CFG.spawn.articulation_props,
)

@configclass
class RobotarmSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # 1. Ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # 2. Workpiece
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/eunseop/isaac/isaac_save/flat_surface_2.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 3. Robot (핵심 수정 부분)
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # [수정 1] 바닥 충돌 방지를 위해 공중으로 20cm 띄움
            pos=(0.0, 0.0, 0.2), 
            # [수정 2] 태어날 때부터 자세를 지정 (Joint Names는 USD 표준 이름 가정)
            # 만약 이름이 틀려도 에러 없이 무시되므로 안전합니다.
            joint_pos={
                ".*": 0.0, # 기본 0도
                "shoulder_lift_joint": -1.57, # 어깨 들기
                "elbow_joint": 1.57,          # 팔꿈치 굽히기
                "wrist_1_joint": -1.57,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # [수정 3] 로봇을 흐물거리게 해서 튕김 방지 (Stiffness 낮춤)
                stiffness=100.0,  
                damping=50.0,    
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
        scale=0.05, 
        use_default=True, 
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 복잡한 관측값 잠시 제외하고 기본만 유지 가능
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
    """Reward terms."""
    # 에러가 날 수 있는 복잡한 리워드는 가중치 0으로 두거나 주석 처리
    track_path = RewTerm(func=local_rew.track_path_reward, weight=10.0, params={"sigma": 0.1})
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=10.0)
    
    # 충돌하면 튕기므로 잠시 0.0으로 둡니다
    collision_penalty = RewTerm(
        func=local_rew.pen_table_collision, 
        weight=0.0, # <-- 안전 모드
        params={"threshold": 0.05} 
    )

@configclass
class EventCfg:
    # [핵심 수정 4] 커스텀 리셋 함수 제거 -> 표준 함수 사용
    # init_state에서 이미 자세를 잡았으므로 reset은 기본값으로 가도 됩니다.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    pass

@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=128, env_spacing=2.5)
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
