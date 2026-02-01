# 파일 경로: RobotArm/tasks/manager_based/robotarm/robotarm_env_cfg.py

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

# [모듈 임포트]
import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew      

# 로봇 모델
from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# -------------------------------------------------------------------------
# Scene Configuration
# -------------------------------------------------------------------------

TEMP_ROBOT_CFG = copy.deepcopy(UR10E_W_SPINDLE_CFG)
TEMP_ROBOT_CFG.spawn.activate_contact_sensors = True  # 센서 필수 활성화

@configclass
class RobotarmSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # 1. 바닥 (Ground)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # 2. 작업물 (Workpiece)
    # [설정] 작업물의 윗면 높이가 중요합니다.
    # 만약 평평한 판이라면 pos Z를 조절하여 표면 높이를 정의하세요.
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/eunseop/isaac/isaac_save/flat_surface_2.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.7, 0.0, 0.0), # 로봇 앞 0.7m 지점
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 3. 로봇 (Robot)
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        # [핵심] 초기 자세 설정 (Cobra Pose - 내려다보는 자세)
        # 이 설정이 없으면 로봇이 누워있거나 펴진 상태로 시작해 휘적거립니다.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,    # 정면
                "shoulder_lift_joint": -2.0,  # 팔을 뒤로 들어올림
                "elbow_joint": 2.0,           # 팔꿈치를 앞으로 굽힘
                "wrist_1_joint": -1.57,       # 손목을 아래로 꺾음 (-90도)
                "wrist_2_joint": -1.57,       # 툴 회전 정렬 (-90도)
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=400.0, 
                damping=40.0, 
            ),
        }
    )

    # 4. 접촉 센서
    contact_forces = ContactSensorCfg(
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
class ActionsCfg:
    """Action specifications."""
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.05, # 세밀한 제어를 위해 스케일 축소
        use_default=True, # 초기 자세(init_state)를 기준으로 움직임
    )

@configclass
class ObservationsCfg:
    """Observation specifications."""
    @configclass
    class PolicyCfg(ObsGroup):
        # [Custom] 경로 추종 정보
        path_tracking = ObsTerm(func=local_obs.path_tracking_obs)
        # [Default] 관절 정보
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # [Custom] EE 자세 정보 (자세 학습에 필수)
        ee_history = ObsTerm(func=local_obs.ee_pose_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms."""
    
    # 1. 경로 추종 (기본 목표)
    track_path = RewTerm(
        func=local_rew.track_path_reward, 
        weight=5.0, 
        params={"sigma": 0.1}
    )
    
    # 2. [중요] 자세 유지 (수직 하강 자세)
    # 가중치를 높여서 자세가 안 맞으면 점수를 못 받게 함
    orientation = RewTerm(
        func=local_rew.rew_tool_orientation, 
        weight=10.0
    )
    
    # 3. [중요] 표면 밀착 (공중부양 방지)
    # 작업물 높이가 0.05m(5cm) 정도라고 가정. 실제 USD 높이에 맞춰 수정 필요.
    surface_contact = RewTerm(
        func=local_rew.rew_surface_tracking,
        weight=5.0,
        params={"target_height": 0.05} 
    )

    # 4. [중요] 충돌 방지 (뚫음 방지)
    # 작업물 높이(0.05)보다 내려가면 감점.
    collision_penalty = RewTerm(
        func=local_rew.pen_table_collision,
        weight=20.0, # 매우 큰 가중치 (양수면 func에서 음수 리턴해야 함. 여기선 func가 음수 리턴하므로 weight는 양수)
        params={"threshold": 0.05}
    )
    
    # 5. 부드러운 움직임
    smoothness = RewTerm(
        func=local_rew.action_smoothness_penalty, 
        weight=-0.05
    )


@configclass
class EventCfg:
    """Configuration for events."""
    
    # [핵심] 에피소드 시작 시 로봇 자세 리셋
    # mdp.reset_joints_by_offset은 Scene에 정의된 init_state(Cobra Pose)를 기준으로
    # 약간의 노이즈만 섞어서 초기화합니다. 즉, 시작부터 작업 자세를 잡습니다.
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
    """Termination terms."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 충돌이 너무 심하면(깊이 뚫으면) 에피소드 종료
    # illegal_contact = DoneTerm(...) # 필요 시 추가


@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the polishing robot arm environment."""
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=128, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 10.0 # 학습 초반엔 짧게 가져가는 게 좋음
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
