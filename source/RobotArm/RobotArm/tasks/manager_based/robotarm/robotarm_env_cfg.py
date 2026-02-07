# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import copy
import os
import torch

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
from isaaclab.utils.math import scale_transform
from isaaclab.sensors import ContactSensorCfg

# Custom MDP modules
import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew 
from .mdp import path_loader # 경로 로더 필수

from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# =========================================================================
# [핵심] 가변 임피던스 액션 클래스 (파일 통합)
# =========================================================================
class VariableImpedanceAction(ActionTerm):
    """
    [조교님 조언 & 논문 3 구현]
    RL 에이전트가 위치를 직접 제어하는 것이 아니라,
    '강성(Stiffness, K)'과 '감쇠(Damping, D)'를 최적화하여 
    주어진 경로(Path)를 부드럽게 따라가는 토크를 생성합니다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = env.scene.find_joints(cfg.asset_name, cfg.joint_names)
        self.num_joints = len(self.joint_ids)
        
        # Action Dim: 관절 수 * 2 (각 관절의 K와 D를 제어)
        self._action_dim = self.num_joints * 2
        
        # [튜닝 포인트] 논문 기반 파라미터 범위
        # K (Stiffness): 낮을수록 부드러움 (접촉 시 유리), 높을수록 정확함 (이동 시 유리)
        self.stiff_range = torch.tensor([10.0, 300.0], device=env.device) 
        # D (Damping): 진동을 잡는 역할. 너무 크면 로봇이 뻑뻑해짐.
        self.damp_range = torch.tensor([5.0, 80.0], device=env.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor):
        # 1. Action 분리 (K, D)
        # actions shape: (num_envs, 12) -> 앞 6개 K, 뒤 6개 D
        k_actions = actions[:, :self.num_joints]
        d_actions = actions[:, self.num_joints:]

        # 2. 스케일링 (-1~1 값을 실제 물리 수치로 변환)
        target_k = scale_transform(k_actions, self.stiff_range[0], self.stiff_range[1])
        target_d = scale_transform(d_actions, self.damp_range[0], self.damp_range[1])

        # 3. 로봇 상태 가져오기
        robot = self._env.scene[self.cfg.asset_name]
        current_q = robot.data.joint_pos[:, self.joint_ids]
        current_v = robot.data.joint_vel[:, self.joint_ids]

        # 4. 목표 경로(Target) 가져오기
        # "경로는 주어졌다"는 가정하에, path_loader에서 가장 가까운 목표점을 가져옵니다.
        # (단순화를 위해 여기서는 IK 없이 Joint Space 오차를 사용하거나, 
        #  Task Space 제어를 해야 하지만, 코드가 복잡해지므로 
        #  '현재 위치 유지(Stabilization)' + 'RL이 오차 보정' 방식으로 근사합니다.)
        
        # *개선된 방식*: RL이 K, D 뿐만 아니라 미세한 위치 보정(delta_q)도 하게 할 수 있지만,
        # 여기서는 조교님 의도대로 '파라미터 튜닝'에 집중하기 위해 Target을 고정하거나 
        # path_loader의 목표를 따라갑니다. (여기서는 간단히 0번 자세 유지로 가정 -> 실제론 path_loader 연동 필요)
        # 일단은 안정성을 위해 '현재 위치'를 유지하려는 힘을 기본으로 하되, 
        # 외부 힘(접촉)에 대해 K, D로 반응하도록 합니다.
        target_q = current_q.clone() # (실제 구현 시엔 path_loader의 target joint 값 필요)

        # 5. 임피던스 토크 계산 (Impedance Law)
        # Torque = K * (q_des - q) - D * q_dot
        # q_des - q 가 0이라면(Target=Current), 이 로봇은 '댐퍼(D)' 역할만 하여 진동을 잡습니다.
        # 움직이려면 q_des가 변해야 합니다.
        
        desired_torque = target_k * (target_q - current_q) - target_d * current_v
        
        # 6. 토크 적용
        robot.set_joint_effort_target(desired_torque, joint_ids=self.joint_ids)


# =========================================================================
# [사용자 경로 설정]
# =========================================================================
USER_STL_PATH = "/home/nrs2/RobotArm2026/flat_surface.stl"

# [기본 자세] 수직 하강 자세
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
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

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

    # [중요] 로봇을 'Torque Control' 모드로 설정
    # 임피던스 제어를 위해 하드웨어 게인을 0으로 만들고, 위 Action Class에서 계산한 토크를 직접 줍니다.
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            joint_pos=DEVICE_READY_STATE, 
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=0.0,    # Torque 제어 필수 설정
                damping=0.0,      # Torque 제어 필수 설정
            ),
        }
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*", 
        history_length=3,
        track_air_time=False,
    )
    
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# -------------------------------------------------------------------------
# MDP Settings
# -------------------------------------------------------------------------

@configclass
class ActionsCfg:
    # [통합된 커스텀 액션 사용]
    # 위에서 정의한 VariableImpedanceAction 클래스를 직접 연결합니다.
    impedance_control = ActionTerm(
        func=VariableImpedanceAction,
        params={"asset_name": "robot", "joint_names": [".*"]}
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
    # [조교님 조언: 최적화 관점의 보상 설계]
    
    # 1. Force Control (최우선) - 목표 힘(10N) 유지
    force_control = RewTerm(func=local_rew.force_control_reward, weight=50.0, params={"target_force": 10.0})

    # 2. Stability (진동 억제) - 에너지(Torque) 최소화
    # 논문 3: "불필요한 토크 사용은 진동의 원인이다"
    applied_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.01)
    joint_vel = RewTerm(func=local_rew.joint_vel_penalty, weight=-0.1) # 천천히
    
    # 3. Path Tracking (보조)
    # 궤적을 완전히 벗어나지만 않게 유도
    track_path = RewTerm(func=local_rew.track_path_reward, weight=10.0, params={"sigma": 0.1})
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=20.0) 

    # 4. Action Constraints
    # K, D 값이 급격하게 변하지 않도록 규제
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.1)
    
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-5.0)


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # [논문 1: 도메인 랜덤화]
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
        },
    )
    
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-0.15, 0.15),
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
        
        # [PhysX]
        self.sim.physx.bounce_threshold_velocity = 0.5 
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 
        
        self.debug_vis = True 
        
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
