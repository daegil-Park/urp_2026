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

# [핵심 수정 1] 로봇 자체에 CCD(연속 충돌 감지)를 켜서 '터널링' 방지
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
        max_depenetration_velocity=1.0, # 뚫고 들어갔을 때 튕겨나오는 속도 제한
        enable_ccd=True, # [중요] 고속 이동 시 벽 뚫기 방지
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

    # 2. Workpiece (물리 재질 추가)
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.1), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)), # 어두운 회색
            
            # [핵심 수정 2] 물리 재질 (Physics Material) 추가
            # 마찰력을 높여서 로봇이 닿았을 때 미끄러지지 않고 '단단함'을 느끼게 함
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  # 정지 마찰
                dynamic_friction=1.0, # 운동 마찰
                restitution=0.0,      # 반발 계수 (0 = 튀어오르지 않음, 흡수)
            ),
            
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, # 고정된 벽
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(), 
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.05),
        ),
    )

    # 3. Robot (제어 파라미터 튜닝)
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            joint_pos=DEVICE_READY_STATE, 
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # [핵심 수정 3] 강성(Stiffness)은 낮추고 댐핑(Damping)은 높임
                # 너무 딱딱하면(Stiffness High) 충돌 시 발광함.
                # 댐핑이 높으면 물속에 있는 것처럼 움직임이 차분해짐.
                stiffness=100.0,   
                damping=60.0,     
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
        # [핵심 수정 4] Action Scale 축소
        # 로봇이 한 번에 너무 크게 움직이려 하면 물리 엔진이 깨짐.
        # 0.02 -> 0.01로 줄여서 더 조심스럽게 움직이도록 함.
        scale=0.01, 
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

    # Penalties (움직임 억제 강화)
    # 속도와 가속도에 대한 페널티를 강화하여 휘젓기 방지
    joint_vel = RewTerm(func=local_rew.joint_vel_penalty, weight=-0.05) 
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.05)
    
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
    
    # 바닥 뚫기 방지 (Safety Net)
    underground_death = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.02, "asset_cfg": SceneEntityCfg("robot")}
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
        # [핵심 수정 5] 물리 엔진 주사율(Frequency) 상향
        # dt를 줄여서 물리 계산을 더 촘촘하게 수행 -> 터널링 방지에 결정적
        self.sim.dt = 1.0 / 120.0 
        self.sim.render_interval = self.decimation
        
        # [PhysX 설정 강화]
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024
        
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
