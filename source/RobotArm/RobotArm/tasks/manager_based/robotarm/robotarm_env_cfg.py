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
# [Action] 디버깅 로그가 포함된 하이브리드 액션
# =========================================================================
class HybridPolishingAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = env.scene.find_joints(cfg.asset_name, cfg.joint_names)
        self.num_joints = len(self.joint_ids)
        self._action_dim = 12 
        
        # State: 0(Align), 1(Descend), 2(Magnet Polish)
        self.state = torch.zeros(env.num_envs, dtype=torch.int, device=env.device) 
        self.timer = torch.zeros(env.num_envs, device=env.device)
        self.path_timer = torch.zeros(env.num_envs, device=env.device)
        self.contact_z = torch.zeros(env.num_envs, device=env.device)

        self.k_pos_range = torch.tensor([10.0, 1500.0], device=env.device)
        self.k_rot_range = torch.tensor([100.0, 1000.0], device=env.device) 
        self.d_range = torch.tensor([10.0, 150.0], device=env.device)

        self.center_x = 0.75
        self.center_y = 0.0
        
        # [DEBUG] 실행 확인용 플래그
        self.debug_printed = False

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor):
        # [DEBUG] 이 로그가 안 뜨면 파일 연결이 잘못된 것임
        if not self.debug_printed:
            print("\n" + "="*50)
            print("🚀 [HybridPolishingAction] LOADED SUCCESSFULLY!")
            print("   State 0: Align -> State 1: Vertical Descend -> State 2: Magnet")
            print("="*50 + "\n")
            self.debug_printed = True
            
        dt = self._env.step_dt
        self.timer += dt
        
        # 1. Robot & Sensor State
        robot = self._env.scene[self.cfg.asset_name]
        ee_pos = robot.data.body_pos_w[:, -1, :]   
        ee_quat = robot.data.body_quat_w[:, -1, :] 
        
        sensor = self._env.scene.sensors["contact_forces"]
        # Z축 힘 (절댓값)
        force_z = torch.abs(sensor.data.net_forces_w[..., 2]).max(dim=-1)[0]

        # 2. Target Init (수직 쿼터니언 고정)
        target_pos = ee_pos.clone()
        # [0, 1, 0, 0]이 UR10e Base 기준 바닥을 보는 자세라고 가정
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self._env.device).repeat(self._env.num_envs, 1)

        # RL Params
        k_pos = scale_transform(actions[:, 0:3], self.k_pos_range[0], self.k_pos_range[1])
        k_rot = scale_transform(actions[:, 3:6], self.k_rot_range[0], self.k_rot_range[1])
        d_pos = scale_transform(actions[:, 6:9], self.d_range[0], self.d_range[1])
        d_rot = scale_transform(actions[:, 9:12], self.d_range[0], self.d_range[1])

        F_bias = torch.zeros_like(ee_pos)

        # ------------------------------------------------------------------
        # State 0: Align (상공 20cm 정지)
        # ------------------------------------------------------------------
        mask_align = (self.state == 0)
        if torch.any(mask_align):
            target_pos[mask_align, 0] = self.center_x
            target_pos[mask_align, 1] = self.center_y
            target_pos[mask_align, 2] = 0.20 
            
            k_pos[mask_align] = 2000.0
            k_rot[mask_align] = 2000.0 
            d_pos[mask_align] = 100.0

            err = torch.norm(target_pos[mask_align] - ee_pos[mask_align], dim=-1)
            ready = (self.timer > 2.0) & (err < 0.02)
            
            switch_ids = torch.nonzero(mask_align).flatten()[ready]
            if len(switch_ids) > 0:
                self.state[switch_ids] = 1
                self.timer[switch_ids] = 0.0
                # print(f"[DEBUG] Env {switch_ids[0]} -> State 1 (Descend)")
        
        # ------------------------------------------------------------------
        # State 1: Descend (수직 하강)
        # ------------------------------------------------------------------
        mask_descend = (self.state == 1)
        if torch.any(mask_descend):
            target_pos[mask_descend, 0] = self.center_x
            target_pos[mask_descend, 1] = self.center_y
            # 매 스텝 0.5mm 하강
            target_pos[mask_descend, 2] = ee_pos[mask_descend, 2] - 0.0005
            
            k_pos[mask_descend] = 1000.0
            k_rot[mask_descend] = 2000.0
            
            # 접촉 감지 (2N)
            contacted = (force_z > 2.0)
            switch_ids = torch.nonzero(mask_descend).flatten()[contacted[mask_descend]]
            
            if len(switch_ids) > 0:
                self.state[switch_ids] = 2
                self.contact_z[switch_ids] = ee_pos[switch_ids, 2]
                self.timer[switch_ids] = 0.0
                self.path_timer[switch_ids] = 0.0
                # print(f"[DEBUG] Env {switch_ids[0]} -> Contact at {ee_pos[switch_ids[0], 2]:.4f}m")

        # ------------------------------------------------------------------
        # State 2: Magnet Polishing (자석 모드)
        # ------------------------------------------------------------------
        mask_polish = (self.state == 2)
        if torch.any(mask_polish):
            self.path_timer[mask_polish] += dt
            t = self.path_timer[mask_polish]
            
            # 자석 효과: 지하 2cm 목표
            target_z = self.contact_z[mask_polish] - 0.02
            
            # ㄹ자 경로
            path_x = self.center_x + 0.15 * torch.sin(0.2 * t)
            path_y = 0.2 * torch.sin(3.0 * t)
            
            target_pos[mask_polish, 0] = path_x
            target_pos[mask_polish, 1] = path_y
            target_pos[mask_polish, 2] = target_z
            
            k_rot[mask_polish] = torch.clamp(k_rot[mask_polish], min=500.0)
            
            # 자석 효과: Bias Force (-20N)
            F_bias[mask_polish, 2] = -20.0 

        # 3. OSC Calculation
        pos_err = target_pos - ee_pos
        quat_inv = quat_conjugate(ee_quat)
        q_diff = quat_mul(target_quat, quat_inv)
        rot_err = 2.0 * torch.sign(q_diff[:, 0]).unsqueeze(1) * q_diff[:, 1:]
        
        vel_lin = robot.data.body_vel_w[:, -1, :3]
        vel_ang = robot.data.body_vel_w[:, -1, 3:]
        
        F_pos = (k_pos * pos_err - d_pos * vel_lin) + F_bias
        F_rot = k_rot * rot_err - d_rot * vel_ang
        
        F_task = torch.cat([F_pos, F_rot], dim=-1)
        
        jacobian = robot.data.jacobian_w[:, self.joint_ids, :]
        j_t = jacobian.transpose(-2, -1)
        desired_torque = torch.bmm(j_t, F_task.unsqueeze(-1)).squeeze(-1)
        
        robot.set_joint_effort_target(desired_torque, joint_ids=self.joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = 0
        self.timer[env_ids] = 0.0
        self.path_timer[env_ids] = 0.0


# =========================================================================
# Scene Config
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
    # 물리 강화된 박스 바닥
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True,
                solver_position_iteration_count=16, solver_velocity_iteration_count=8,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, 0.0)),
    )
    # 힘 제한된 로봇
    robot: ArticulationCfg = UR10E_W_SPINDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=DEVICE_READY_STATE),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"], effort_limit=150.0, velocity_limit=100.0,
                stiffness=0.0, damping=2.0,
            ),
        }
    )
    robot.spawn.rigid_props.enable_ccd = True 
    robot.spawn.rigid_props.solver_position_iteration_count = 16

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*", history_length=3, track_air_time=False,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# =========================================================================
# MDP Config
# =========================================================================
@configclass
class ActionsCfg:
    # [중요] 이름을 'arm_action'으로 원복하여 기존 시스템과 호환성 유지
    arm_action = ActionTerm(
        func=HybridPolishingAction, 
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
    force_tracking = RewTerm(func=local_rew.force_tracking_reward, weight=100.0, params={"target_force": 10.0})
    orientation_align = RewTerm(func=local_rew.orientation_align_reward, weight=80.0, params={"target_axis": (0,0,-1)})
    track_path = RewTerm(func=local_rew.track_path_reward, weight=30.0, params={"sigma": 0.1})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-10.0)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=local_rew.bad_orientation_termination, params={"limit_angle": 0.78})

@configclass
class EventCfg:
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
        self.sim.substeps = 2
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.enable_stabilization = True
