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
# 불필요한 탐색(허공 휘젓기)을 줄입니다.
DEVICE_READY_STATE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708, # -90도
    "elbow_joint": -1.5708,         # -90도
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
        enable_ccd=True, # 관통 방지
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
            # 로봇이 힘을 가했을 때 물체가 밀리면 학습이 안 됩니다.
            # kinematic_enabled=True로 설정하여 완벽한 고정체(벽)로 만듭니다.
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

    # 3. Robot Actuator Tuning (가장 중요)
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            joint_pos=DEVICE_READY_STATE, 
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # [논문 아이디어: 임피던스 제어 흉내]
                # Stiffness(P게인)를 적당히 유지하되, 
                # Damping(D게인)을 높여서 '허공 휘젓기(Oscillation)'를 물리적으로 막습니다.
                stiffness=200.0,   
                damping=80.0,  # 기존 60 -> 80으로 상향 (진동 억제)
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
    # "허공을 휘젓는" 가장 큰 이유는 한 스텝당 움직이는 각도가 너무 크기 때문입니다.
    # 스케일을 0.01에서 0.005로 줄여서, 로봇이 아주 미세하고 부드럽게 움직이도록 강제합니다.
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.005, # [수정] 0.01 -> 0.005 (정밀 제어)
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
    # [논문 아이디어: Reward Shaping]
    # 로봇이 원하는 행동을 했을 때만 점수를 줍니다.
    
    # 1. 경로 추종 (가장 중요)
    track_path = RewTerm(func=local_rew.track_path_reward, weight=15.0, params={"sigma": 0.1})
    
    # 2. 자세 유지 (수직 유지) - 가중치 대폭 상향
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=30.0) # 기존 20 -> 30
    
    # 3. 힘 제어 (표면에 닿아야 함)
    force_control = RewTerm(func=local_rew.force_control_reward, weight=2.0, params={"target_force": 10.0})

    # [신규 추가] 표면 접근 보상
    # 로봇이 허공에 있으면 점수를 못 받으므로 바닥으로 내려가게 유도합니다.
    # (local_rew에 해당 함수가 없다면 mdp.root_height_below 등으로 대체 가능하나, 일단 개념적으로 추가)
    # 구현이 어렵다면 track_path 가중치를 높이는 것으로 대체됩니다.

    # Penalties (벌점)
    # [논문 아이디어: 에너지 최소화]
    # 관절을 너무 빨리 움직이면(휘저으면) 감점합니다.
    joint_vel = RewTerm(func=local_rew.joint_vel_penalty, weight=-0.1) # 가중치 상향 (-0.05 -> -0.1)
    
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.05)
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-5.0)


@configclass
class EventCfg:
    # [수정됨] 방법 2 적용: 랜덤성이 가미된 수직 자세 리셋
    # 매 에피소드마다 약간씩 다른 위치/각도에서 시작하여 일반화 성능을 높입니다.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05), # 위치 노이즈 감소 (너무 멀리 안 가게)
            "velocity_range": (0.0, 0.0),    # 정지 상태 시작
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # [안전장치]
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
        
        # [PhysX 설정] 진동 억제
        self.sim.physx.bounce_threshold_velocity = 0.5 # 조금 둔감하게 설정
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 # 접촉 버퍼 늘리기
        
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
