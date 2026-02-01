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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ViewerCfg

# [중요] 기본 MDP와 커스텀 MDP 모듈 임포트
# (파일 구조가 tasks/manager_based/robotarm/ 안에 있다고 가정)
import isaaclab.envs.mdp as mdp  # Isaac Lab 기본 MDP
from .mdp import observations as local_obs # 우리가 만든 observations.py
from .mdp import rewards as local_rew     # 우리가 만든 rewards.py

# 로봇 모델 임포트
from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# -------------------------------------------------------------------------
# Scene Configuration
# -------------------------------------------------------------------------

# [Robot Config 수정] - Stiffness/Damping 튜닝 및 센서 활성화
TEMP_ROBOT_CFG = copy.deepcopy(UR10E_W_SPINDLE_CFG)

# 기존 USD 경로 유지
original_usd_path = UR10E_W_SPINDLE_CFG.spawn.usd_path 

# spawn 객체 재정의 (Contact Sensor 활성화 필수)
TEMP_ROBOT_CFG.spawn = sim_utils.UsdFileCfg(
    usd_path=original_usd_path,
    activate_contact_sensors=True, # [중요] True로 설정해야 힘 제어 가능
    rigid_props=UR10E_W_SPINDLE_CFG.spawn.rigid_props,
    articulation_props=UR10E_W_SPINDLE_CFG.spawn.articulation_props,
)

@configclass
class RobotarmSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # 1. 바닥 (Ground)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # 2. 작업물 (Workpiece) - 경로 추종 대상
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/eunseop/isaac/isaac_save/flat_surface_2.usd" # 사용자 지정 경로
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 3. 로봇 (Robot)
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), # 필요시 0.05 등으로 수정
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,  # -90도
                "elbow_joint": 1.57,           # 90도
                "wrist_1_joint": -1.57,        # -90도
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
            },
        ),
        # [튜닝] 관절 강성 조절 (진동 방지)
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=400.0,  # 강하게 지탱
                damping=80.0,     # 진동 흡수
            ),
        }
    )

    # 4. 접촉 센서 (Contact Sensor) - 힘 제어 보상용
    contact_forces = ContactSensorCfg(
        # 로봇의 모든 링크에 대한 접촉 감지
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*", 
        history_length=3,
        track_air_time=False,
    )
    
    # 5. 조명
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# -------------------------------------------------------------------------
# MDP Settings
# -------------------------------------------------------------------------

@configclass
class CommandsCfg:
    """Command specifications."""
    # 외부 경로(Path Loader)를 사용하므로 랜덤 명령 생성 불필요
    pass


@configclass
class ActionsCfg:
    """Action specifications."""
    # Joint Position Control
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ],
        scale=0.05, # [튜닝] 작은 값으로 설정하여 부드러운 움직임 유도
        use_default=True, # 초기 위치 기준 상대 제어
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # [Custom] 1. 경로 추종 오차 + 힘 센서 값 (7 dims)
        # observations.py의 path_tracking_obs 함수 연결
        path_tracking = ObsTerm(func=local_obs.path_tracking_obs)
        
        # [Default] 2. 로봇 관절 상태
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # [Custom] 3. 엔드이펙터 히스토리 (History Len * 7 dims)
        # observations.py의 ee_pose_history 함수 연결
        ee_history = ObsTerm(func=local_obs.ee_pose_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms."""
    
    # [Custom] 1. 경로 추종 (Main Objective)
    # rewards.py의 track_path_reward 연결
    track_path = RewTerm(func=local_rew.track_path_reward, weight=10.0, params={"sigma": 0.1})
    
    # [Custom] 2. 힘 제어 (Force Control)
    # rewards.py의 force_control_reward 연결
    force_control = RewTerm(func=local_rew.force_control_reward, weight=2.0, params={"target_force": 10.0})
    
    # [Custom] 3. 자세 유지 (Orientation Alignment)
    # rewards.py의 orientation_align_reward 연결
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=5.0)
    
    # [Custom] 4. 동작 부드러움 (Action Smoothness Penalty)
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.1)
    
    # [Default] 5. 작업 영역 이탈 방지
    # (rewards.py에 새로 만든 out_of_bounds_penalty 사용)
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-5.0)


@configclass
class EventCfg:
    """Configuration for events."""
    # 에피소드 시작 시 로봇 관절 초기화
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05), # 약간의 랜덤성 부여
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms."""
    # 시간 초과 시 종료
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 과도한 힘 발생 시 종료 (충돌 감지)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact, 
        params={
            "threshold": 150.0, # 100N -> 150N으로 약간 완화 (작업 중 오인식 방지)
            "sensor_cfg": SceneEntityCfg("contact_forces") # Scene에 정의된 이름
        }
    )
    
    # [Optional] 키보드 입력('K')으로 강제 종료 (디버깅용)
    # user_reset = DoneTerm(func=local_obs.manual_termination)


@configclass
class CurriculumCfg:
    """Curriculum terms."""
    # 현재는 사용 안 함 (필요 시 주석 해제)
    pass


# -------------------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------------------

@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the polishing robot arm environment."""

    # 1. Scene Settings
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=128, env_spacing=2.5)
    
    # 2. Basic Components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # 3. MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # 시뮬레이션 설정
        self.decimation = 4            # 제어 주기 비율 (Low Level 120Hz / 4 = 30Hz Control)
        self.episode_length_s = 15.0   # 에피소드 길이
        
        # Viewer 설정
        self.viewer.eye = [2.0, 2.0, 2.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        
        # Physics 설정 (120Hz로 부드럽게)
        self.sim.dt = 1.0 / 120.0 
        self.sim.render_interval = self.decimation
        
        # [Custom Info] 작업 영역 크기 (Reward 계산 등에 활용 가능)
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
