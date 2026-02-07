# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import copy
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
from isaaclab.utils.math import scale_transform, quat_from_euler_xyz, quat_mul, quat_conjugate, quat_rotate
from isaaclab.sensors import ContactSensorCfg

# Custom MDP modules
import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew 

from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# =========================================================================
# [핵심] ㄹ자 경로 폴리싱 액션 (Operational Space Control)
# =========================================================================
class RasterPolishingAction(ActionTerm):
    """
    [로봇청소기 모드]
    1. 수학적으로 'ㄹ자(Raster Scan)' 경로를 생성합니다. (경로 파일 대체용)
    2. Operational Space Control(OSC)을 사용하여 EE가 경로를 따라가게 합니다.
    3. RL은 이 제어기의 강성(Stiffness)과 감쇠(Damping)를 튜닝합니다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = env.scene.find_joints(cfg.asset_name, cfg.joint_names)
        self.num_joints = len(self.joint_ids)
        
        # Action Dim: Stiffness(6) + Damping(6) = 12
        # (XYZ 위치 강성 + RX,RY,RZ 회전 강성 / XYZ 감쇠 + 회전 감쇠)
        self._action_dim = 12 
        
        # [튜닝] RL이 조절할 물성치 범위
        # 폴리싱은 위치는 단단하게(High K), 회전은 조금 유연하게, 접촉은 부드럽게(High D)
        self.stiff_range = torch.tensor([50.0, 1000.0], device=env.device) 
        self.damp_range = torch.tensor([10.0, 150.0], device=env.device)
        
        self.time_idx = torch.zeros(env.num_envs, device=env.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt
        self.time_idx += dt
        
        # 1. RL Action -> Impedance Gains (K, D)
        # actions: (num_envs, 12)
        # XYZ+Rot Stiffness / XYZ+Rot Damping
        Kp = scale_transform(actions[:, :6], self.stiff_range[0], self.stiff_range[1])
        Kd = scale_transform(actions[:, 6:], self.damp_range[0], self.damp_range[1])

        # 2. [ㄹ자 경로 생성] (Raster Scan Path)
        # "나중에는 경로 파일을 넣겠지만, 지금은 수식으로 만듦"
        # 중심점: (0.75, 0.0, 0.05)
        # X축: 천천히 전진 (Velocity = 2cm/s)
        # Y축: 빠르게 왕복 (Frequency = 0.5Hz, Amplitude = 15cm)
        
        center_x = 0.75
        center_z = 0.03 # 표면 살짝 위 (3cm) - 누르는 힘은 RL이 K 조절로 결정
        
        # X: -0.15 ~ +0.15 범위를 왕복
        path_x = center_x + 0.15 * torch.sin(0.2 * self.time_idx) 
        # Y: -0.2 ~ +0.2 범위를 빠르게 왕복 (ㄹ자)
        path_y = 0.2 * torch.sin(3.0 * self.time_idx)
        path_z = torch.full_like(path_x, center_z)
        
        target_pos = torch.stack([path_x, path_y, path_z], dim=-1)

        # 3. [자세 생성] 무조건 바닥(수직) 보기
        # UR10e 기준: EE Z축이 World -Z를 봐야 함 (또는 X축 회전 -180도)
        # Quaternion for looking down (Euler: 180, 0, 0 or similar depending on UR convention)
        # 여기서는 World Frame 기준 Fixed Orientation
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self._env.device).repeat(self._env.num_envs, 1) 
        # (Note: 쿼터니언 값은 로봇의 기본 자세에 따라 [0, 1, 0, 0]이 180도 회전일 수 있음. 확인 필요)

        # 4. 로봇 상태 (Jacobian & EE State)
        robot = self._env.scene[self.cfg.asset_name]
        # Jacobian: (num_envs, 6, num_joints)
        jacobian = robot.data.jacobian_w[:, self.joint_ids, :] 
        
        ee_pos = robot.data.body_pos_w[:, -1, :]
        ee_quat = robot.data.body_quat_w[:, -1, :]
        ee_vel_lin = robot.data.body_vel_w[:, -1, :3]
        ee_vel_ang = robot.data.body_vel_w[:, -1, 3:]

        # 5. 오차 계산 (Task Space Error)
        # Position Error
        pos_error = target_pos - ee_pos
        
        # Orientation Error (Quaternion Difference)
        # q_err = q_des * q_curr_inv
        quat_inv = quat_conjugate(ee_quat)
        q_diff = quat_mul(target_quat, quat_inv)
        # Convert to rotation vector (Axis-Angle) approximation
        # q = [w, x, y, z]. If w~1, angle is small. vec ~= 2*[x,y,z]
        # (Sign flip check needed for w < 0)
        sign = torch.sign(q_diff[:, 0]).unsqueeze(1)
        rot_error = 2.0 * sign * q_diff[:, 1:] # (num_envs, 3)

        # Total Error Vector (6D)
        error_task = torch.cat([pos_error, rot_error], dim=-1) # (num_envs, 6)
        velocity_task = torch.cat([ee_vel_lin, ee_vel_ang], dim=-1) # (num_envs, 6)

        # 6. Operational Space Force Calculation
        # F_task = Kp * Error - Kd * Velocity
        # (RL이 Kp, Kd를 조절하여 표면을 '얼마나 세게 누를지' 결정)
        F_task = Kp * error_task - Kd * velocity_task

        # 7. Joint Torque Mapping (Jacobian Transpose)
        # Tau = J^T * F_task
        # (num_envs, num_joints, 6) @ (num_envs, 6, 1) -> (num_envs, num_joints)
        jacobian_T = jacobian.transpose(-2, -1)
        desired_torque = torch.bmm(jacobian_T, F_task.unsqueeze(-1)).squeeze(-1)

        # 8. 토크 인가
        robot.set_joint_effort_target(desired_torque, joint_ids=self.joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.time_idx[env_ids] = 0.0


# =========================================================================
# Scene & Env Config
# =========================================================================

USER_STL_PATH = "/home/nrs2/RobotArm2026/flat_surface.stl"

# [수직 자세] 
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

    # [중요] Torque Control Mode
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
                stiffness=0.0,    # Action에서 OSC로 계산하므로 0
                damping=0.0,      
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
    # [교체] ㄹ자 경로 + OSC 제어 액션
    polishing_motion = ActionTerm(
        func=RasterPolishingAction,
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
    # [폴리싱 최적화 보상]
    
    # 1. Force Control (최우선)
    # 10N의 힘으로 지긋이 눌러야 함
    force_control = RewTerm(func=local_rew.force_control_reward, weight=50.0, params={"target_force": 10.0})

    # 2. Orientation (필수)
    # 작업 중에 로봇 손목이 꺾이거나 비틀어지면 안 됨 (수직 유지)
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=40.0) 

    # 3. Path Tracking
    # ㄹ자 경로를 잘 따라가는지
    track_path = RewTerm(func=local_rew.track_path_reward, weight=15.0, params={"sigma": 0.1})

    # 4. Stability
    # 진동 방지
    applied_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.01)
    joint_vel = RewTerm(func=local_rew.joint_vel_penalty, weight=-0.1)
    
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-10.0)


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
        
        self.sim.physx.bounce_threshold_velocity = 0.5 
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 
        
        self.debug_vis = True 
        
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
