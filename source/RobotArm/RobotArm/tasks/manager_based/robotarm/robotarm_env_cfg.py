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
# [수직 자세] 기본 자세 (논문 아이디어: 초기화 전략)
# -------------------------------------------------------------------------
# 로봇이 시작할 때부터 '작업 준비 자세(Ready Pose)'를 취하게 하여
# 불필요한 탐색 시간을 줄이고, 초기 충돌을 방지합니다.
DEVICE_READY_STATE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708, # -90도 (팔을 세움)
    "elbow_joint": -1.5708,         # -90도 (앞으로 굽힘)
    "wrist_1_joint": -1.5708,       # -90도 (손목을 아래로)
    "wrist_2_joint": 1.5708,        # +90도 (툴 정렬)
    "wrist_3_joint": 0.0,
}

# -------------------------------------------------------------------------
# Scene Configuration
# -------------------------------------------------------------------------

TEMP_ROBOT_CFG = copy.deepcopy(UR10E_W_SPINDLE_CFG)

# [로봇 물리 설정]
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
        enable_ccd=True, # [중요] 관통(Tunneling) 방지
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
            
            # [논문 아이디어: 강체 역학]
            # kinematic_enabled=True로 설정하여 완벽한 고정체(벽)로 만듭니다.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0, # 튕김 없음
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.0), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 3. Robot Actuator Tuning (진동 억제)
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            joint_pos=DEVICE_READY_STATE, 
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # [튜닝 포인트]
                # Damping을 80.0으로 높여서 '물속에서 움직이는 듯한' 저항감을 줍니다.
                # 이는 허공을 휘젓는(Flailing) 현상을 물리적으로 막습니다.
                stiffness=200.0,   
                damping=80.0,  
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
    # [논문 아이디어: Action Scaling]
    # 스케일을 0.005로 줄여 로봇의 움직임을 미세하게 만듭니다.
    # 이는 강화학습 초기의 불안정한 탐색으로 인한 '급발진'을 막습니다.
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.005, 
        use_default=True, 
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # [논문 1 적용: POMDP 해결을 위한 관측 강화]
        # 현재 위치뿐만 아니라 속도(Velocity)와 과거 이력(History)을 모두 줍니다.
        path_tracking = ObsTerm(func=local_obs.path_tracking_obs)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # [중요] 과거 5프레임의 데이터를 묶어서 제공 (History Stacking)
        ee_history = ObsTerm(func=local_obs.ee_pose_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # [논문 아이디어: Reward Shaping]
    
    # 1. 경로 추종 (Main Task)
    track_path = RewTerm(func=local_rew.track_path_reward, weight=15.0, params={"sigma": 0.1})
    
    # 2. 자세 유지 (수직) - 가중치 대폭 상향
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=30.0) 
    
    # 3. 힘 제어 (Contact)
    force_control = RewTerm(func=local_rew.force_control_reward, weight=3.0, params={"target_force": 10.0})

    # 4. [신규] 표면 접근 (Approach)
    # 로봇이 바닥 근처(3cm)까지 과감하게 내려오도록 유도
    approach = RewTerm(func=local_rew.surface_approach_reward, weight=5.0, params={"target_height": 0.03})

    # Penalties
    # [논문 1 적용: 에너지 효율성] 불필요한 고속 동작 감점
    joint_vel = RewTerm(func=local_rew.joint_vel_penalty, weight=-0.1)
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.05)
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-5.0)


@configclass
class EventCfg:
    # 1. 초기화 전략 (Method 2)
    # 매 에피소드마다 '수직 자세' 근처에서 랜덤하게 시작
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # [논문 1 적용: 도메인 랜덤화 (Domain Randomization)]
    # 시뮬레이션과 현실의 간극(Sim-to-Real Gap)을 줄이기 위한 핵심입니다.
    
    # 2. 마찰력 랜덤화 (미끄러짐 변화 학습)
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),  # ±20%
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
        },
    )

    # 3. 질량 랜덤화 (무게 오차 학습)
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-0.1, 0.1), # ±10%
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # [안전장치] 지하로 뚫고 내려가면 종료
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
        
        # [PhysX 설정] 물리 안정성 최적화
        self.sim.physx.bounce_threshold_velocity = 0.5 
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 
        
        # [논문 2 적용: 시각적 가이드]
        # 학습 중 디버깅을 위해 시각화 마커 활성화
        self.debug_vis = True 
        
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
