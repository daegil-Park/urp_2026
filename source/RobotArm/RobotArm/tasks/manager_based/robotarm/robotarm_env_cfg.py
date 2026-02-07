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

# =========================================================================
# [사용자 경로 설정]
# =========================================================================
USER_STL_PATH = "/home/nrs2/RobotArm2026/flat_surface.stl"

# -------------------------------------------------------------------------
# [기본 자세] Ready Pose
# -------------------------------------------------------------------------
DEVICE_READY_STATE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": -1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": 1.5708,
    "wrist_3_joint": 0.0,
}

# -------------------------------------------------------------------------
# Scene Configuration
# -------------------------------------------------------------------------

TEMP_ROBOT_CFG = copy.deepcopy(UR10E_W_SPINDLE_CFG)

TEMP_ROBOT_CFG.spawn = sim_utils.UsdFileCfg(
    usd_path=UR10E_W_SPINDLE_CFG.spawn.usd_path,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0, 
        enable_ccd=True, 
    ),
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
            usd_path=USER_STL_PATH,  
            scale=(1.0, 1.0, 1.0),   
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0, 
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.0), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 3. Robot Actuator Tuning (조교님 조언 반영: Admittance 느낌)
    # 기존보다 Stiffness를 낮추고(150), Damping을 높여(100) 
    # 로봇 자체를 '스프링-댐퍼' 시스템처럼 만듭니다. (논문 3의 핵심)
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            joint_pos=DEVICE_READY_STATE, 
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # [수정] Stiffness 감소 (200 -> 150): 너무 딱딱하면 힘 제어가 안 됨
                # [수정] Damping 증가 (80 -> 100): 진동을 더 강력하게 억제
                stiffness=150.0,   
                damping=100.0,  
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
    # [조교님 조언: 주어진 경로]
    # 여기에 '이상적인 경로(Reference Trajectory)'를 정의하는 커맨드를 넣을 수 있습니다.
    # 지금은 Rewards에서 수학적으로 계산하지만, 추후 여기에 Path Generator를 붙일 수 있습니다.
    pass

@configclass
class ActionsCfg:
    # [조교님 조언: 세부 파라미터 개선]
    # Action Scale을 극도로 줄여서(0.001), RL이 '경로 생성'을 하는 게 아니라
    # '경로 오차 보정(Fine-tuning)'만 수행하도록 제한합니다.
    # 이것이 사실상 "Admittance Control"의 RL 버전입니다. (Force -> Position Offset)
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.001, # [핵심 변경] 매우 작게 설정
        use_default=True, 
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # [논문 1, 3 적용] 상태 관측
        path_tracking = ObsTerm(func=local_obs.path_tracking_obs)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # [중요] 힘 제어를 하려면 '과거의 힘 데이터'를 아는 것이 필수입니다.
        # History에 힘 센서 값도 포함되면 좋지만, 일단 EE Pose History로 간접 추정합니다.
        ee_history = ObsTerm(func=local_obs.ee_pose_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # [조교님 조언 + 논문 3: Reward 구조 변경]
    # "경로를 따라가라"는 이미 쉬운 문제이므로 가중치를 낮춥니다.
    # "힘을 맞춰라(Force Control)"와 "진동하지 마라(Stability)"에 집중합니다.
    
    # 1. Force Control (최우선 목표)
    # 목표 힘(10N)을 맞추면 점수를 가장 크게 줍니다.
    force_control = RewTerm(func=local_rew.force_control_reward, weight=50.0, params={"target_force": 10.0})

    # 2. Path Tracking (기본 목표)
    # 경로에서 이탈하지 않는지 체크 (가중치 15 -> 10 하향)
    track_path = RewTerm(func=local_rew.track_path_reward, weight=10.0, params={"sigma": 0.1})
    
    # 3. Orientation (자세 유지)
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=20.0) 
    
    # 4. Stability (진동 억제)
    # [논문 3 적용] 에너지(Torque) 최소화 = 부드러운 움직임
    applied_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.01)
    
    # 급격한 행동 변화(Jerk) 방지
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.1)
    
    # 속도 제한 (천천히 작업)
    joint_vel = RewTerm(func=local_rew.joint_vel_penalty, weight=-0.1)

    # 5. Fail Cases
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-5.0)


@configclass
class EventCfg:
    # [논문 1 적용] Sim-to-Real Robustness
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 물리 파라미터 랜덤화 (필수)
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
        },
    )
    
    # 질량 랜덤화 (어드미턴스 제어의 M 파라미터 대응 훈련)
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-0.1, 0.1),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    underground_death = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=64, env_spacing=2.5)
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
        
        # [안정성 설정]
        self.sim.physx.bounce_threshold_velocity = 0.5 
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 
        
        self.debug_vis = True 
        
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
