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
# [수정 1] 회전은 단단하게, 위치는 부드럽게 잡는 OSC 액션
# =========================================================================
class RobustPolishingAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = env.scene.find_joints(cfg.asset_name, cfg.joint_names)
        self.num_joints = len(self.joint_ids)
        
        # Action Dim: 12 (Pos Stiffness 3, Rot Stiffness 3, Pos Damping 3, Rot Damping 3)
        # RL이 위치와 회전을 분리해서 제어하도록 변경
        self._action_dim = 12 
        
        self.time_idx = torch.zeros(env.num_envs, device=env.device)
        
        # [튜닝] 회전(Rot) 강성을 위치(Pos)보다 훨씬 높게 설정
        self.k_pos_range = torch.tensor([10.0, 1000.0], device=env.device)
        self.k_rot_range = torch.tensor([50.0, 500.0], device=env.device) # 회전은 기본적으로 단단하게
        self.damp_range = torch.tensor([10.0, 100.0], device=env.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt
        self.time_idx += dt
        
        # 1. Parameter Decoding
        # RL 출력 -> 물리 파라미터 변환
        k_pos = scale_transform(actions[:, 0:3], self.k_pos_range[0], self.k_pos_range[1])
        k_rot = scale_transform(actions[:, 3:6], self.k_rot_range[0], self.k_rot_range[1]) # 회전 강성
        d_pos = scale_transform(actions[:, 6:9], self.damp_range[0], self.damp_range[1])
        d_rot = scale_transform(actions[:, 9:12], self.damp_range[0], self.damp_range[1])
        
        # [긴급 처방] 수직 유지를 위해 회전 강성을 강제로 높임 (RL이 낮추지 못하게)
        k_rot = torch.clamp(k_rot, min=200.0) 

        # 2. Raster Scan Trajectory (ㄹ자 경로)
        center_x = 0.75
        center_z = 0.045 # [중요] 표면(0.05)보다 살짝 아래로 설정해야 '누르는 힘'이 생김
        
        path_x = center_x + 0.15 * torch.sin(0.2 * self.time_idx) 
        path_y = 0.2 * torch.sin(3.0 * self.time_idx)
        path_z = torch.full_like(path_x, center_z)
        
        target_pos = torch.stack([path_x, path_y, path_z], dim=-1)
        
        # 3. Target Orientation (무조건 바닥 보기)
        # UR10e에서 EE가 바닥을 보려면 쿼터니언이 특정 값이어야 함.
        # (시뮬레이션 좌표계에 따라 [0, 1, 0, 0] 또는 [1, 0, 0, 0] 등 확인 필요)
        # 여기서는 Identity([1,0,0,0])가 아니라 "바닥을 보는 회전"을 줘야 함.
        # 임시로: 현재 로봇의 초기 자세(Ready Pose)의 회전값을 Target으로 고정
        robot = self._env.scene[self.cfg.asset_name]
        
        # *팁: Ready State일 때의 쿼터니언을 하드코딩해서 박아버리는 게 제일 확실함
        # UR10e Base 기준 바닥보기: 보통 Rotate X by 180 deg
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self._env.device).repeat(self._env.num_envs, 1)

        # 4. Error Calculation
        ee_pos = robot.data.body_pos_w[:, -1, :]
        ee_quat = robot.data.body_quat_w[:, -1, :]
        
        pos_err = target_pos - ee_pos
        
        # Orientation Error
        quat_inv = quat_conjugate(ee_quat)
        q_diff = quat_mul(target_quat, quat_inv)
        rot_err = 2.0 * torch.sign(q_diff[:, 0]).unsqueeze(1) * q_diff[:, 1:]
        
        # 5. Force Calculation (F = Kx - Dv)
        vel_lin = robot.data.body_vel_w[:, -1, :3]
        vel_ang = robot.data.body_vel_w[:, -1, 3:]
        
        F_pos = k_pos * pos_err - d_pos * vel_lin
        F_rot = k_rot * rot_err - d_rot * vel_ang # 여기서 k_rot가 높아야 수직 유지
        
        F_task = torch.cat([F_pos, F_rot], dim=-1)
        
        # 6. Jacobian Mapping
        jacobian = robot.data.jacobian_w[:, self.joint_ids, :]
        j_t = jacobian.transpose(-2, -1)
        desired_torque = torch.bmm(j_t, F_task.unsqueeze(-1)).squeeze(-1)
        
        robot.set_joint_effort_target(desired_torque, joint_ids=self.joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.time_idx[env_ids] = 0.0


# =========================================================================
# Scene Config (물리적 충돌 강화)
# =========================================================================
USER_STL_PATH = "/home/nrs2/RobotArm2026/flat_surface.stl"
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

    # [수정 2] STL 대신 확실한 '박스'로 바닥 만들기 (테스트용)
    # STL 파일의 메쉬 문제일 수 있으므로, 일단 Isaac 기본 박스로 물리 충돌을 보장합니다.
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.1), # 가로 세로 1m, 두께 10cm 박스
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, # 고정된 벽
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)), # 초록색 박스
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.0), # Z=0이면 박스 윗면은 Z=0.05가 됨 (중심 기준이므로)
        ),
    )

    robot: ArticulationCfg = UR10E_W_SPINDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=DEVICE_READY_STATE),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"], effort_limit=300.0, velocity_limit=100.0,
                stiffness=0.0, damping=2.0, # [수정] 약간의 댐핑을 줘서 떨림 방지
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
# Config 연결
# =========================================================================
@configclass
class ActionsCfg:
    polishing = ActionTerm(func=RobustPolishingAction, params={"asset_name": "robot", "joint_names": [".*"]})
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
    # 1. Force (10N)
    force_tracking = RewTerm(func=local_rew.force_tracking_reward, weight=80.0, params={"target_force": 10.0})
    # 2. Orientation (수직)
    orientation_align = RewTerm(func=local_rew.orientation_align_reward, weight=50.0, params={"target_axis": (0,0,-1)})
    # 3. Path
    track_path = RewTerm(func=local_rew.track_path_reward, weight=20.0, params={"sigma": 0.1})
    # 4. Penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-10.0)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # [수정] 수직에서 45도 이상 꺾이면 바로 종료 (학습 가속화)
    bad_orientation = DoneTerm(func=local_rew.bad_orientation_termination, params={"limit_angle": 0.78}) 

@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg() # (기존 유지)
    commands: CommandsCfg = CommandsCfg() # (기존 유지)
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.enable_stabilization = True
