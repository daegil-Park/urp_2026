# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
import math
import copy

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ViewerCfg

import isaaclab.envs.mdp as mdp

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
# [수정 코드] - spawn 설정을 완전히 새로 만듭니다.
TEMP_ROBOT_CFG = copy.deepcopy(UR10E_W_SPINDLE_CFG)

# 기존 USD 경로 가져오기 (만약 에러나면 직접 경로 문자열을 넣으세요)
original_usd_path = UR10E_W_SPINDLE_CFG.spawn.usd_path 

# spawn 객체를 새로 생성하여 강제 할당
TEMP_ROBOT_CFG.spawn = sim_utils.UsdFileCfg(
    usd_path=original_usd_path,
    activate_contact_sensors=True,  # 여기서 확실하게 켭니다
    rigid_props=UR10E_W_SPINDLE_CFG.spawn.rigid_props,
    articulation_props=UR10E_W_SPINDLE_CFG.spawn.articulation_props,
)

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

    # [수정] 위에서 만든 TEMP_ROBOT_CFG를 사용합니다.
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        # 1. 초기 자세 수정 (엔드이펙터가 표적 근처에 오도록 설정)
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05), # z축으로 5cm만 띄워보세요 (기존 0.0)
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.0,  # 팔을 앞으로 내림
                "elbow_joint": 1.5,           # 팔꿈치를 굽힘
                "wrist_1_joint": -2.0,       # 손목 수평 맞춤
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
            },
        ),
        # 2. 관절 강성(Stiffness) 조절 (떨림 방지 핵심!)
        # 기존 로봇 설정이 너무 딱딱해서(Stiffness 높음) 덜덜 떨리는 것입니다.
        # 값을 부드럽게(100.0) 낮추고, 저항(Damping 40.0)을 줘서 진동을 잡습니다.
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"], # 모든 관절에 적용
                stiffness=100.0,         # 강성 (높으면 떨림, 낮으면 처짐)
                damping=40.0,            # 감쇠 (진동 흡수)
            ),
        }
    )

    # [NEW] 접촉 센서 추가 (로봇의 모든 링크 감지)
    contact_forces = ContactSensorCfg(
        # 기존: "{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*"
        # 수정: 중간에 /Robot/ 을 포함하거나, 구체적인 링크를 지정
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/Robot/.*", 
        history_length=3,
        track_air_time=False,
    )
    
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
            
    policy: PolicyCfg = PolicyCfg()


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
    
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default, 
        mode="reset",
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # [FIXED] 충돌 감지 설정 수정
    # threshold와 함께 sensor_cfg를 반드시 넣어줘야 함
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact, 
        params={
            "threshold": 100.0,
            "sensor_cfg": SceneEntityCfg("contact_forces") # SceneCfg에 정의한 이름과 일치해야 함
        }
    )

  #  user_reset = DoneTerm(func=local_rew.manual_termination)

##
# Environment configuration
##

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # coverage_curriculum = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "coverage",
    #         "weight": -0.0004,
    #         "num_steps": 10000}
    # )

    # out_of_bounds_curriculum = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "out_of_bounds_penalty",
    #         "weight": 0.0002,
    #         "num_steps": 5000}
    # )

    # ee_orientation_curriculum = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "ee_orientation_alignment",
    #         "weight": 0.0003,
    #         "num_steps":7000}
    # )

    # time_efficiency_curriculum = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "time_efficiency",
    #         "weight": 0.0001,
    #         "num_steps": 10000}
    # )



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
        self.decimation = 4 # 제어 주기는 30Hz 유지 (120 / 4 = 30)
        self.episode_length_s = 15.0 # 에피소드 길이 (15초 동안 폴리싱)
        
        # 뷰어 설정 (카메라 위치)
        self.viewer.eye = [2.0, 2.0, 2.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        # simulation settings
        # [수정] 물리 계산을 1초에 120번 하도록 변경 (더 부드러워짐)
        self.sim.dt = 1.0 / 120.0 
        self.sim.render_interval = self.decimation
        
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
