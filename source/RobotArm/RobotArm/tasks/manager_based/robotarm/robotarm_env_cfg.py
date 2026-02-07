# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import copy
import torch
import numpy as np

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
from isaaclab.utils.math import scale_transform, quat_mul, quat_conjugate
from isaaclab.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew 

from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# =========================================================================
# [1] 가상 경로 생성기 (The "Green Line")
# =========================================================================
class PolishingPathGenerator:
    """
    사용자가 언급한 '초록색 선'을 수학적으로 생성합니다.
    나중에 실제 waypoint 파일이 생기면 이 클래스만 교체하면 됩니다.
    """
    def __init__(self, device, num_envs):
        self.device = device
        self.num_envs = num_envs
        
        # 작업물 표면 기준 (Workpiece Center: 0.75, 0.0, 0.0)
        # 높이(Z)는 표면(0.05)보다 살짝 아래(0.048)로 설정하여
        # Stiffness에 의해 자연스럽게 10N이 눌리도록 유도 (Penetration Depth)
        self.surface_z = 0.05 
        self.target_z = 0.048 
        self.center_x = 0.75
        
        # 경로 파라미터 (ㄹ자 형태)
        self.width_y = 0.2   # 좌우 폭
        self.length_x = 0.15 # 진행 길이
        self.scan_freq = 2.0 # 좌우 왕복 속도

    def get_target_pose(self, t_scan):
        """
        t_scan: Polishing Phase가 시작된 후 흐른 시간
        return: target_pos (x,y,z), target_quat
        """
        # X축: 천천히 전진
        # Y축: 빠르게 왕복 (Sinewave) -> 이것이 '초록색 선'의 형태
        path_x = self.center_x - (self.length_x / 2.0) + 0.02 * t_scan 
        path_y = self.width_y * torch.sin(self.scan_freq * t_scan)
        path_z = torch.full_like(path_x, self.target_z)
        
        target_pos = torch.stack([path_x, path_y, path_z], dim=-1)
        
        # Orientation: 무조건 바닥(수직)을 보라.
        # UR10e Base 기준, EE가 바닥을 보려면 적절한 회전 필요
        # (여기서는 단순화를 위해 Identity quaternion 가정, 로봇 세팅에 따라 수정 필요)
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        
        return target_pos, target_quat

    def get_start_pose(self):
        """ 빨간 점 (Start Point) 위치 반환 """
        # t=0 일 때의 위치
        return self.get_target_pose(torch.zeros(self.num_envs, device=self.device))


# =========================================================================
# [2] 통합 폴리싱 액션 (Phase Logic 포함)
# =========================================================================
class PolishingMissionAction(ActionTerm):
    """
    [미션]
    1. Start Point(빨간 점)로 이동 (Approach)
    2. 접촉 후 안정화 (Stabilize)
    3. 초록색 경로 추종 (Polishing) - 이때 RL이 파라미터(K, D) 최적화
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = env.scene.find_joints(cfg.asset_name, cfg.joint_names)
        self.num_joints = len(self.joint_ids)
        self._action_dim = 12 # (Stiffness 6 + Damping 6)
        
        self.path_gen = PolishingPathGenerator(env.device, env.num_envs)
        
        # 내부 상태 관리
        self.timer = torch.zeros(env.num_envs, device=env.device)
        self.phase = torch.zeros(env.num_envs, dtype=torch.int, device=env.device) 
        # 0: Approach, 1: Stabilize, 2: Polishing
        
        # 파라미터 범위 (RL이 조절할 범위)
        self.k_range = torch.tensor([10.0, 2000.0], device=env.device)
        self.d_range = torch.tensor([5.0, 200.0], device=env.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt
        self.timer += dt
        
        # 1. RL Action 해석 (Stiffness, Damping)
        # RL은 항상 최적의 K, D를 내놓고 있다고 가정
        kp_val = scale_transform(actions[:, :6], self.k_range[0], self.k_range[1])
        kd_val = scale_transform(actions[:, 6:], self.d_range[0], self.d_range[1])
        
        # 2. Phase 별 목표 위치 설정
        target_pos_start, target_quat_start = self.path_gen.get_start_pose()
        
        # 로봇 현재 상태
        robot = self._env.scene[self.cfg.asset_name]
        ee_pos = robot.data.body_pos_w[:, -1, :]
        ee_quat = robot.data.body_quat_w[:, -1, :]
        jacobian = robot.data.jacobian_w[:, self.joint_ids, :]
        
        # [Phase Logic]
        # 간단한 시간 기반 페이즈 전환 (실제로는 센서 접촉 여부로 하면 더 좋음)
        approach_time = 2.0
        stabilize_time = 1.0
        
        target_pos = torch.zeros_like(ee_pos)
        target_quat = torch.zeros_like(ee_quat)
        
        # Phase 0: Approach (빨간 점 위 1cm에서 빨간 점으로 부드럽게 하강)
        mask_approach = self.timer < approach_time
        if torch.any(mask_approach):
            # approach 중에는 시작점보다 살짝 위 -> 시작점으로 이동
            ratio = self.timer[mask_approach] / approach_time
            # 보간: (Start Z + 0.1) -> (Start Z)
            target_pos[mask_approach] = target_pos_start[mask_approach]
            target_pos[mask_approach, 2] += 0.05 * (1 - ratio) # 5cm 위에서 내려옴
            target_quat[mask_approach] = target_quat_start[mask_approach]
            
            # 접근 중에는 RL 파라미터 대신 안전한 기본값 사용 (튕김 방지)
            kp_val[mask_approach] = 500.0
            kd_val[mask_approach] = 50.0

        # Phase 1: Stabilize (빨간 점에서 정지)
        mask_stable = (self.timer >= approach_time) & (self.timer < approach_time + stabilize_time)
        if torch.any(mask_stable):
            target_pos[mask_stable] = target_pos_start[mask_stable]
            target_quat[mask_stable] = target_quat_start[mask_stable]
            # 안정화 중에도 조금 단단하게 잡음
            kp_val[mask_stable] = 800.0
            kd_val[mask_stable] = 80.0

        # Phase 2: Polishing (초록색 선 따라가기 - RL이 개입하는 진짜 구간)
        mask_polish = self.timer >= (approach_time + stabilize_time)
        if torch.any(mask_polish):
            # Polishing 시작 후 경과 시간
            t_polish = self.timer[mask_polish] - (approach_time + stabilize_time)
            p_pos, p_quat = self.path_gen.get_target_pose(t_polish)
            target_pos[mask_polish] = p_pos
            target_quat[mask_polish] = p_quat
            # 이때는 RL이 출력한 kp_val, kd_val이 그대로 적용됨 (최적화 대상)

        # 3. Operational Space Control (OSC) 토크 계산
        # Error Calculation
        pos_err = target_pos - ee_pos
        
        # Orientation Error
        quat_inv = quat_conjugate(ee_quat)
        q_diff = quat_mul(target_quat, quat_inv)
        rot_err = 2.0 * torch.sign(q_diff[:, 0]).unsqueeze(1) * q_diff[:, 1:]
        
        error_task = torch.cat([pos_err, rot_err], dim=-1)
        vel_task = torch.cat([robot.data.body_vel_w[:, -1, :3], robot.data.body_vel_w[:, -1, 3:]], dim=-1)
        
        # F = K * err - D * vel
        F_task = kp_val * error_task - kd_val * vel_task
        
        # Jacobian Transpose to Joint Torque
        j_t = jacobian.transpose(-2, -1)
        desired_torque = torch.bmm(j_t, F_task.unsqueeze(-1)).squeeze(-1)
        
        robot.set_joint_effort_target(desired_torque, joint_ids=self.joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.timer[env_ids] = 0.0


# =========================================================================
# [3] Scene & Config
# =========================================================================

USER_STL_PATH = "/home/nrs2/RobotArm2026/flat_surface.stl"
DEVICE_READY_STATE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": -1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": 1.5708,
    "wrist_3_joint": 0.0,
}

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
            usd_path=USER_STL_PATH, scale=(1.0, 1.0, 1.0),   
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # Torque Control Mode (Stiffness=0)
    robot: ArticulationCfg = UR10E_W_SPINDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=DEVICE_READY_STATE),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"], effort_limit=300.0, velocity_limit=100.0,
                stiffness=0.0, damping=0.0, # Pure Torque Control
            ),
        }
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*", history_length=3, track_air_time=False,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# =========================================================================
# [4] MDP - Rewards (The Critic)
# =========================================================================

@configclass
class ActionsCfg:
    polishing_mission = ActionTerm(func=PolishingMissionAction, params={"asset_name": "robot", "joint_names": [".*"]})
    gripper_action: ActionTerm | None = None

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # RL이 봐야할 정보: 현재 위치, 힘, 속도, 지난 액션
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        force_sensor = ObsTerm(func=local_obs.force_sensor_reading) # (구현 필요 or contact_forces 사용)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    # --- 핵심 목표 ---
    
    # 1. [Force] 10N 유지 (가장 중요)
    # 10N 근처일 때 점수 높고, 벗어나면 급격히 감점
    force_tracking = RewTerm(
        func=local_rew.force_tracking_reward, 
        weight=100.0, 
        params={"target_force": 10.0, "tolerance": 1.0}
    )

    # 2. [Orientation] 수직 유지 (필수)
    # Spindle Z축이 World -Z축과 정렬되어야 함
    orientation_align = RewTerm(
        func=local_rew.orientation_align_reward, 
        weight=50.0,
        params={"target_axis": (0,0,-1)}
    )

    # 3. [Path] 경로 추종 (Contact 상태에서 벗어나지 말 것)
    # EE가 목표 지점(Green Line)에서 멀어지면 벌점
    path_tracking = RewTerm(func=local_rew.track_path_reward, weight=20.0, params={"sigma": 0.05})

    # --- 보조 목표 ---
    
    # 4. [Smoothness] 파라미터가 미친듯이 튀지 않도록
    action_smoothness = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    
    # 5. [Safety] 진동 방지
    joint_vel_limit = RewTerm(func=mdp.joint_vel_l2, weight=-0.1)

    # 실패 조건 벌점
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    # 너무 멀리 벗어나거나 뒤집어지면 에피소드 종료
    episode_timeout = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=local_rew.bad_orientation_termination, params={"limit_angle": 0.5})

@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg() # (기존 코드 유지)
    commands: CommandsCfg = CommandsCfg() # (기존 코드 유지)
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0 # 접근+안정화(3초) + 폴리싱(17초)
        self.sim.dt = 1.0 / 120.0
        self.sim.physx.bounce_threshold_velocity = 0.2 # 튕김 최소화 설정
        self.sim.physx.enable_stabilization = True
