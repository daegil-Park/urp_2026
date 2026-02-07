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
from isaaclab.utils.math import scale_transform, quat_mul, quat_conjugate
from isaaclab.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew 

from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# =========================================================================
# [핵심] 하이브리드 액션 (Rule-based Init + RL Optimization)
# =========================================================================
class HybridPolishingAction(ActionTerm):
    """
    [사용자 요청 반영] JacobianController.py 로직 이식
    
    State 0 (Approach): 시작점 상공에서 수직 자세 정렬 (High Stiffness)
    State 1 (Landing): 아주 천천히 하강하며 접촉 감지 (Soft Landing)
    State 2 (RL Polishing): 접촉 후 RL 파라미터로 ㄹ자 경로 추종
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = env.scene.find_joints(cfg.asset_name, cfg.joint_names)
        self.num_joints = len(self.joint_ids)
        self._action_dim = 12 # RL Output: Pos_K(3), Rot_K(3), Pos_D(3), Rot_D(3)
        
        # 내부 상태 변수
        self.state = torch.zeros(env.num_envs, dtype=torch.int, device=env.device) 
        # 0: Approach, 1: Landing, 2: Polishing
        self.timer = torch.zeros(env.num_envs, device=env.device)
        self.contact_z = torch.zeros(env.num_envs, device=env.device) # 접촉 지점 높이 저장

        # [튜닝] RL 파라미터 범위
        self.k_pos_range = torch.tensor([10.0, 1500.0], device=env.device)
        self.k_rot_range = torch.tensor([100.0, 1000.0], device=env.device) # 회전은 기본적으로 단단하게
        self.d_range = torch.tensor([10.0, 150.0], device=env.device)

        # 경로 생성용 변수
        self.center_x = 0.75
        self.path_timer = torch.zeros(env.num_envs, device=env.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt
        self.timer += dt
        
        # --- 1. 센서 데이터 및 로봇 상태 ---
        robot = self._env.scene[self.cfg.asset_name]
        ee_pos = robot.data.body_pos_w[:, -1, :]
        ee_quat = robot.data.body_quat_w[:, -1, :]
        jacobian = robot.data.jacobian_w[:, self.joint_ids, :]
        
        # 힘 센서 (Z축 힘) 가져오기
        sensor = self._env.scene.sensors["contact_forces"]
        force_z = torch.abs(sensor.data.net_forces_w[..., 2]).max(dim=-1)[0] # (num_envs,)

        # --- 2. State Machine (보내주신 코드 로직 구현) ---
        
        # 목표값 초기화
        target_pos = ee_pos.clone()
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self._env.device).repeat(self._env.num_envs, 1) # 수직(바닥)

        # RL 파라미터 디코딩 (기본값)
        k_pos = scale_transform(actions[:, 0:3], self.k_pos_range[0], self.k_pos_range[1])
        k_rot = scale_transform(actions[:, 3:6], self.k_rot_range[0], self.k_rot_range[1])
        d_pos = scale_transform(actions[:, 6:9], self.d_range[0], self.d_range[1])
        d_rot = scale_transform(actions[:, 9:12], self.d_range[0], self.d_range[1])

        # [State 0] Approach (상공 이동 & 자세 정렬)
        mask_approach = (self.state == 0)
        if torch.any(mask_approach):
            # 목표: (0.75, 0.0, 0.15) -> 상공 15cm
            target_pos[mask_approach, 0] = self.center_x
            target_pos[mask_approach, 1] = 0.0
            target_pos[mask_approach, 2] = 0.15 
            
            # 2초 지나면 다음 단계로
            # 강성은 매우 단단하게 (자세 잡기 위해)
            k_pos[mask_approach] = 2000.0
            k_rot[mask_approach] = 1000.0 # [중요] 수직 자세 꽉 잡기
            d_pos[mask_approach] = 100.0
            
            # 거리 오차가 작으면 다음 스테이지
            dist_err = torch.norm(target_pos[mask_approach] - ee_pos[mask_approach], dim=-1)
            ready_envs = (self.timer > 2.0) & (dist_err < 0.01)
            # mask_approach 중 ready_envs인 것들의 state를 1로 변경
            # (인덱싱 주의: 전체 인덱스에서 mask_approach가 True인 것 중 ready가 True인 것)
            switch_ids = torch.nonzero(mask_approach).flatten()[ready_envs]
            self.state[switch_ids] = 1
            self.timer[switch_ids] = 0.0 # 타이머 리셋

        # [State 1] Landing (천천히 하강)
        mask_landing = (self.state == 1)
        if torch.any(mask_landing):
            # XY는 고정, Z만 아주 천천히 내림
            target_pos[mask_landing, 0] = self.center_x
            target_pos[mask_landing, 1] = 0.0
            # 현재 위치보다 0.2mm 아래를 목표로 (속도 제어 효과)
            target_pos[mask_landing, 2] = ee_pos[mask_landing, 2] - 0.0005 
            
            # 강성은 중간 정도 (충격 흡수)
            k_pos[mask_landing] = 800.0
            k_rot[mask_landing] = 1000.0 # 수직은 계속 유지
            
            # 접촉 감지 (2N 이상)
            contacted = (force_z > 2.0)
            switch_ids = torch.nonzero(mask_landing).flatten()[contacted[mask_landing]]
            
            if len(switch_ids) > 0:
                self.state[switch_ids] = 2
                self.contact_z[switch_ids] = ee_pos[switch_ids, 2] # 바닥 높이 기억
                self.timer[switch_ids] = 0.0
                self.path_timer[switch_ids] = 0.0
                # 디버그 출력 대신 로봇이 멈칫하는 걸로 알 수 있음

        # [State 2] Polishing (RL Control + ㄹ자 경로)
        mask_polishing = (self.state == 2)
        if torch.any(mask_polishing):
            self.path_timer[mask_polishing] += dt
            t = self.path_timer[mask_polishing]
            
            # ㄹ자 경로 생성 (초록색 선)
            # Z값: 기억해둔 바닥 높이(contact_z)보다 2mm 아래 (10N 가압 유도)
            target_z = self.contact_z[mask_polishing] - 0.002 
            
            path_x = self.center_x + 0.15 * torch.sin(0.2 * t)
            path_y = 0.2 * torch.sin(3.0 * t)
            
            target_pos[mask_polishing, 0] = path_x
            target_pos[mask_polishing, 1] = path_y
            target_pos[mask_polishing, 2] = target_z
            
            # 이때는 RL이 출력한 k_pos, k_rot 등이 그대로 적용됨 (최적화 대상)
            # 단, 수직 유지를 위해 k_rot 최소값은 보장
            k_rot[mask_polishing] = torch.clamp(k_rot[mask_polishing], min=300.0)

        # --- 3. Operational Space Control (OSC) 토크 계산 ---
        # Position Error
        pos_err = target_pos - ee_pos
        
        # Orientation Error (Quaternion diff)
        quat_inv = quat_conjugate(ee_quat)
        q_diff = quat_mul(target_quat, quat_inv)
        # q_diff의 x,y,z 성분이 회전 오차 축
        rot_err = 2.0 * torch.sign(q_diff[:, 0]).unsqueeze(1) * q_diff[:, 1:]
        
        # Velocity
        vel_lin = robot.data.body_vel_w[:, -1, :3]
        vel_ang = robot.data.body_vel_w[:, -1, 3:]
        
        # Force Command = K*err - D*vel
        F_pos = k_pos * pos_err - d_pos * vel_lin
        F_rot = k_rot * rot_err - d_rot * vel_ang
        
        F_task = torch.cat([F_pos, F_rot], dim=-1)
        
        # Jacobian Transpose Mapping (Task Force -> Joint Torque)
        j_t = jacobian.transpose(-2, -1)
        desired_torque = torch.bmm(j_t, F_task.unsqueeze(-1)).squeeze(-1)
        
        robot.set_joint_effort_target(desired_torque, joint_ids=self.joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = 0 # 리셋 시 Approach 부터 다시 시작
        self.timer[env_ids] = 0.0
        self.path_timer[env_ids] = 0.0


# =========================================================================
# Scene Config (물리 안정성 최우선)
# =========================================================================
USER_STL_PATH = "/home/nrs2/RobotArm2026/flat_surface.stl"
# 수직 자세 (Ready Pose)
DEVICE_READY_STATE = {
    "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.5708, "elbow_joint": -1.5708,
    "wrist_1_joint": -1.5708, "wrist_2_joint": 1.5708, "wrist_3_joint": 0.0,
}

@configclass
class RobotarmSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # [핵심] 초록색 박스 (단단한 바닥, STL 대체)
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.1), # 1m x 1m 바닥
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16, # 물리 계산 4배 강화 (뚫림 방지)
                solver_velocity_iteration_count=8,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, # 5mm 전부터 충돌 감지
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, 0.0)),
    )

    # [핵심] 로봇 설정 (힘 제한)
    robot: ArticulationCfg = UR10E_W_SPINDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=DEVICE_READY_STATE),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"], 
                effort_limit=150.0, # 150Nm로 제한 (뚫고 들어가는 힘 억제)
                velocity_limit=100.0,
                stiffness=0.0, # Torque Mode
                damping=2.0,   # 최소한의 공기 저항
            ),
        }
    )
    # 로봇 자체 충돌 설정 강화
    robot.spawn.rigid_props.enable_ccd = True 
    robot.spawn.rigid_props.solver_position_iteration_count = 16

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*", history_length=3, track_air_time=False,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# =========================================================================
# MDP & Rewards
# =========================================================================
@configclass
class ActionsCfg:
    # 하이브리드 액션 연결
    polishing = ActionTerm(func=HybridPolishingAction, params={"asset_name": "robot", "joint_names": [".*"]})
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
    # 1. Force Tracking (10N 유지) - 가장 중요
    force_tracking = RewTerm(func=local_rew.force_tracking_reward, weight=100.0, params={"target_force": 10.0})
    
    # 2. Orientation (수직 유지)
    orientation_align = RewTerm(func=local_rew.orientation_align_reward, weight=80.0, params={"target_axis": (0,0,-1)})
    
    # 3. Path Tracking (ㄹ자 경로)
    track_path = RewTerm(func=local_rew.track_path_reward, weight=30.0, params={"sigma": 0.1})
    
    # 4. Penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-10.0)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 수직 자세가 45도 이상 무너지면 즉시 종료
    bad_orientation = DoneTerm(func=local_rew.bad_orientation_termination, params={"limit_angle": 0.78})

@configclass
class EventCfg:
    # 초기화 시 약간의 랜덤성 부여 (Robustness)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0)}
    )

@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        # 물리 안정화 설정
        self.sim.substeps = 2
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.enable_stabilization = True
