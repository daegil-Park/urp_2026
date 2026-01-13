# Copyright (c) 2022-2025, The Isaac Lab Project Developers
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


# STL을 바로 못 쓰므로 USD 변환본을 쓰거나 MeshCfg 사용
from isaaclab.sim import UsdFileCfg

# [중요] 기존 mdp 대신 우리가 새로 만든 모듈을 불러옵니다.
# (파일 이름이 바뀌었으므로 경로 수정 필수!)
# 실제 사용 시에는 같은 폴더에 파일을 두고 아래처럼 import 하거나,
# 기존 mdp 폴더 안에 파일들을 넣고 경로를 맞춰주세요.
# 여기서는 편의상 같은 mdp 패키지 내에 있다고 가정합니다.

# 만약 파일을 scripts 폴더 등에 따로 뒀다면 import 경로가 달라질 수 있습니다.
# 일단 기존 구조를 따른다는 가정하에 작성합니다.
# reward, observation modul import
import importlib
local_obs = importlib.import_module("RobotArm.tasks.manager_based.robotarm.mdp.observations")
local_rew = importlib.import_module("RobotArm.tasks.manager_based.robotarm.mdp.rewards")

import RobotArm.tasks.manager_based.robotarm.mdp as mdp # 기본 mdp도 필요할 수 있음 (종료 조건 등)


from RobotArm.robots.ur10e_w_spindle import *

##
# Scene configuration
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
            pos=(0.75, 0.0, 0.0),
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
    """Command specifications for the environment."""
    # 이번 미션은 외부 경로 추종이므로 별도의 랜덤 명령(Command) 생성은 필요 없음
    # (하지만 에러 방지를 위해 null command 등을 남겨둘 수 있음)
    pass


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
        scale=0.05, #스케일만 줄여도 로봇이 발작하는 현상은 대부분 사라집니다.
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # [NEW] 1. 경로 추종 오차 + 힘 센서 값 (핵심)
        path_tracking = ObsTerm(func=local_obs.path_tracking_obs)
        
        # [NEW] 2. 로봇 관절 상태 (기본) - 내가 어떤 자세인지 알아야 함
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # [NEW] 3. 엔드이펙터 히스토리 (속도감, 추세 파악용)
        ee_history = ObsTerm(func=local_obs.ee_pose_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # [NEW] 1. 경로 추종 (가장 중요, 가중치 높음)
    track_path = RewTerm(func=local_rew.track_path_reward, weight=10.0)
    
    # [NEW] 2. 힘 제어 (10N 유지)
    force_control = RewTerm(func=local_rew.force_control_reward, weight=2.0)
    
    # [NEW] 3. 자세 유지 (수직 유지)
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=5.0)
    
    # [NEW] 4. 부드러운 움직임 (Jerk 방지, 벌점)
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.1)
    
    # [Optional] 작업 영역 이탈 시 큰 벌점 (안전장치)
    out_of_bounds = RewTerm(func=mdp.out_of_bounds_penalty, weight=-5.0)

@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0, 0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_grid_mask = EventTerm(
        func=local_rew.reset_grid_mask,
        mode="reset",
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # [기존 유지] 시간이 다 되면 종료
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 충돌/안전 장치
    illegal_contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 100.0})

    # 사용자 강제 종료 (Manual Reset)
    # 키보드 'K'를 누르면 리셋됨
    # 로봇이 경로를 이탈하거나 이상하게 꼬이면 K
    user_reset = DoneTerm(func=local_rew.manual_termination)

##
# Environment configuration
##

@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the polishing robot arm environment."""

    # Scene settings
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=128, env_spacing=2.5) # env 개수는 GPU 메모리에 맞춰 조절
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg() # (기존 파일에 있다면 유지)
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # Curriculum settings (일단 끔, 필요시 켬)
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2 # 시뮬레이션 2번 돌 때 제어 1번 (제어 주기 조절)
        self.episode_length_s = 15.0 # 에피소드 길이 (15초 동안 폴리싱)
        
        # 뷰어 설정 (카메라 위치)
        self.viewer.eye = [2.0, 2.0, 2.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
