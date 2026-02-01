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
import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew      

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
        # [핵심 수정] 초기 자세 변경: 휘적거림 방지를 위해 테이블을 바라보는 자세(Cobra Pose)로 시작
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -2.0,  # 팔을 뒤로 들어올림 (기존 -1.57보다 더 듦)
                "elbow_joint": 2.0,           # 팔꿈치를 앞으로 굽힘 (기존 1.57보다 더 굽힘)
                "wrist_1_joint": -1.57,       # 손목을 아래로 꺾음
                "wrist_2_joint": -1.57,       # 툴 정렬
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
        scale=0.05, # [튜닝] 작은 값 유지 (부드러운 움직임)
        use_default=True, # init_state 기준 상대 제어
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # [Custom] 1. 경로 추종 오차 + 힘 센서 값 (7 dims)
        path_tracking = ObsTerm(func=local_obs.path_tracking_obs)
        
        # [Default] 2. 로봇 관절 상태
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # [Custom] 3. 엔드이펙터 히스토리
        ee_history = ObsTerm(func=local_obs.ee_pose_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms."""
    
    # [Custom] 1. 경로 추종 (Main Objective)
    track_path = RewTerm(func=local_rew.track_path_reward, weight=10.0, params={"sigma": 0.1})
    
    # [Custom] 2. 힘 제어 (Force Control)
    force_control = RewTerm(func=local_rew.force_control_reward, weight=2.0, params={"target_force": 10.0})
    
    # [Custom] 3. 자세 유지 (Orientation Alignment)
    # 기존 함수를 그대로 쓰되, 가중치를 높임
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=10.0)
    
    # [NEW] 4. 충돌 방지 (Table Collision Penalty)
    # 뚫으면 강력한 벌점 (weight는 양수로 두고 함수가 음수를 리턴하므로 작동함)
    collision_penalty = RewTerm(
        func=local_rew.pen_table_collision, 
        weight=50.0, 
        params={"threshold": 0.05} # 작업물 높이 (0.05m 아래로 가면 벌점)
    )

    # [NEW] 5. 표면 밀착 (Surface Tracking)
    # 공중부양 방지
    surface_contact = RewTerm(
        func=local_rew.rew_surface_tracking,
        weight=5.0,
        params={"target_height": 0.05} # 작업물 높이
    )

    # [Custom] 6. 동작 부드러움 (Action Smoothness Penalty)
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.1)
    
    # [Default] 7. 작업 영역 이탈 방지
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-5.0)


@configclass
class EventCfg:
    """Configuration for events."""
    # [수정] 에피소드 시작 시 로봇 관절 초기화
    # init_state(Cobra Pose)를 기준으로 아주 작은 랜덤 오프셋만 적용
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02), # 랜덤 범위를 좁힘 (자세 유지)
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
            "threshold": 200.0, # 뚫고 들어가려고 힘주다가 꺼지는 것 방지 위해 조금 더 완화
            "sensor_cfg": SceneEntityCfg("contact_forces") 
        }
    )


@configclass
class CurriculumCfg:
    """Curriculum terms."""
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
        self.decimation = 4            
        self.episode_length_s = 15.0   
        
        self.viewer.eye = [2.0, 2.0, 2.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        
        self.sim.dt = 1.0 / 120.0 
        self.sim.render_interval = self.decimation
        
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
