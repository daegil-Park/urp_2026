# 파일 경로: RobotArm/tasks/manager_based/robotarm/mdp/rewards.py
from __future__ import annotations

import torch
import math
import os
import csv
import numpy as np
from typing import TYPE_CHECKING

# 시각화 및 로깅 라이브러리 (GUI 충돌 방지)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab.envs import ManagerBasedRLEnv

try:
    import carb.input
except ImportError:
    carb = None

# 경로 생성 모듈 (사용자 정의)
from . import path_loader
import sys
if "nrs_fk_core" not in sys.modules:
    try:
        from nrs_fk_core import FKSolver
    except ImportError:
        FKSolver = None
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -----------------------------------------------------------
# Global Logging Variables
# -----------------------------------------------------------
_path_tracking_history = []
_force_control_history = []
_episode_counter = 0

# -----------------------------------------------------------
# Ideal Joint Pose (수직 작업 자세)
# -----------------------------------------------------------
# 이 값은 robotarm_env_cfg.py의 DEVICE_READY_STATE와 일치해야 합니다.
IDEAL_JOINT_POSE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def get_ee_pose(env: ManagerBasedRLEnv, asset_name: str = "robot"):
    """End-Effector의 World Pose (Position + Quaternion)를 반환"""
    robot = env.scene[asset_name]
    pos = robot.data.body_pos_w[:, -1, :]
    quat = robot.data.body_quat_w[:, -1, :]
    return torch.cat([pos, quat], dim=-1)

# -----------------------------------------------------------
# Reward Functions
# -----------------------------------------------------------

def track_path_reward(env: ManagerBasedRLEnv, sigma: float = 0.1):
    """
    [경로 추종 보상] - 좌표계 보정 적용됨
    """
    global _path_tracking_history, _episode_counter

    # 1. 경로 데이터 가져오기 (Local Frame: 0,0 기준)
    path_tensor = path_loader.get_path_tensor(env.device)
    if path_tensor.device != env.device:
        path_tensor = path_tensor.to(env.device)

    # ---------------------------------------------------------------------
    # [핵심 수정] 좌표계 오프셋 (Coordinate Offset)
    # 작업물(Workpiece)의 위치가 (0.75, 0.0)이므로 경로를 그쪽으로 이동시킵니다.
    # Z값 0.02는 표면보다 살짝 위를 의미합니다.
    # ---------------------------------------------------------------------
    offset = torch.tensor([0.75, 0.0, 0.02], device=env.device)
    
    # 경로를 World 좌표계로 변환 (Broadcasting)
    target_path_world = path_tensor + offset.unsqueeze(0)

    # 2. 현재 End-Effector 위치
    ee_pose = get_ee_pose(env)
    current_pos = ee_pose[:, :3]

    # 3. 거리 계산 (모든 경로 점들과의 거리 중 최소값)
    # (num_envs, 1, 3) - (1, num_points, 3) -> (num_envs, num_points, 3)
    dists = torch.norm(current_pos.unsqueeze(1) - target_path_world.unsqueeze(0), dim=2)
    min_dist, _ = torch.min(dists, dim=1)

    # 4. 로깅 (Logging)
    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _path_tracking_history.append((step, min_dist[0].item()))

    # 에피소드 종료 시 그래프 저장
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots(step)

    # 거리가 가까울수록 보상 증가 (Gaussian Kernel)
    return torch.exp(-torch.square(min_dist) / (sigma ** 2))


def orientation_align_reward(env: ManagerBasedRLEnv):
    """[자세 제어] 수직 방향(Z축 아래) 유지"""
    ee_pose = get_ee_pose(env)
    ee_quat = ee_pose[:, 3:]

    # Tool의 Z축 벡터 (Local)
    tool_z_local = torch.zeros((env.num_envs, 3), device=env.device)
    tool_z_local[:, 2] = 1.0 
    
    # World 좌표계로 변환
    tool_z_world = quat_apply(ee_quat, tool_z_local)
    
    # 목표: World의 -Z 방향 (아래쪽)
    target_dir = torch.zeros_like(tool_z_world)
    target_dir[:, 2] = -1.0 

    # 내적(Dot Product)을 통해 정렬 확인 (1.0이면 완벽 일치)
    dot_prod = torch.sum(tool_z_world * target_dir, dim=-1)
    error = 1.0 - torch.clamp(dot_prod, min=-1.0, max=1.0)
    
    # Scale 30.0으로 강력하게 규제
    return torch.exp(-error * 30.0)


def joint_deviation_reward(env: ManagerBasedRLEnv):
    """[관절 규제] 이상적인 수직 자세에서 벗어나지 않도록"""
    robot = env.scene["robot"]
    current_joints = robot.data.joint_pos 
    
    target_joints = torch.tensor(IDEAL_JOINT_POSE, device=env.device).unsqueeze(0)
    
    diff = torch.norm(current_joints - target_joints, dim=-1)
    return torch.exp(-diff * 0.5)


def force_control_reward(env: ManagerBasedRLEnv, target_force: float = 10.0):
    """[힘 제어] 접촉 시 일정 힘 유지"""
    global _force_control_history
    current_force = torch.zeros(env.num_envs, device=env.device)

    if "contact_forces" in env.scene.sensors:
        sensor = env.scene["contact_forces"]
        # Z축 방향 힘만 고려
        if sensor.data.net_forces_w is not None and sensor.data.net_forces_w.shape[1] > 0:
            force_z = torch.abs(sensor.data.net_forces_w[..., 2])
            current_force, _ = torch.max(force_z, dim=-1)
    
    force_error = torch.abs(current_force - target_force)

    # 로깅
    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _force_control_history.append((step, current_force[0].item(), target_force))

    return 1.0 / (1.0 + 0.1 * force_error)

def joint_vel_penalty(env: ManagerBasedRLEnv):
    """[속도 페널티] 너무 빠르게 움직이거나 휘적거리는 것 방지"""
    robot = env.scene["robot"]
    # 관절 속도의 제곱합에 음수 (비용 함수)
    return -torch.sum(torch.square(robot.data.joint_vel), dim=-1)

def action_smoothness_penalty(env: ManagerBasedRLEnv):
    """[Action 부드러움] 급격한 행동 변화 방지"""
    return -torch.sum(torch.square(env.action_manager.action), dim=-1)

def out_of_bounds_penalty(env: ManagerBasedRLEnv):
    """[작업 영역 이탈 방지]"""
    ee_pos = get_ee_pose(env)[:, :3]
    # 작업대 주변을 벗어나면 페널티
    is_out_x = (ee_pos[:, 0] < -0.1) | (ee_pos[:, 0] > 1.1)
    is_out_y = (ee_pos[:, 1] < -0.6) | (ee_pos[:, 1] > 0.6)
    is_out_z = (ee_pos[:, 2] < 0.0) | (ee_pos[:, 2] > 0.8)
    is_out = (is_out_x | is_out_y | is_out_z).float()
    return -1.0 * is_out

# -----------------------------------------------------------
# Visualization Logic
# -----------------------------------------------------------
def save_episode_plots(step: int):
    global _path_tracking_history, _force_control_history, _episode_counter
    
    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    os.makedirs(save_dir, exist_ok=True)

    if _path_tracking_history:
        steps, dists = zip(*_path_tracking_history)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, dists, label="Distance to Path", color="blue")
        plt.title(f"Path Tracking Error (Ep {_episode_counter + 1})")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"path_error_ep{_episode_counter + 1}.png"))
        plt.close()

    if _force_control_history:
        steps, currents, targets = zip(*_force_control_history)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, currents, label="Current Force", color="red")
        plt.plot(steps, targets, "--", label="Target Force", color="green")
        plt.title(f"Force Control (Ep {_episode_counter + 1})")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"force_control_ep{_episode_counter + 1}.png"))
        plt.close()

    _path_tracking_history.clear()
    _force_control_history.clear()
    _episode_counter += 1
    print(f"[{step}] Plots saved for Episode {_episode_counter}")

# -----------------------------------------------------------
# Reset Logic
# -----------------------------------------------------------
def reset_robot_to_cobra(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    [리셋 로직] 수직 자세(IDEAL_JOINT_POSE)로 초기화
    """
    robot = env.scene["robot"]
    
    target_pose = torch.tensor(IDEAL_JOINT_POSE, device=env.device)
    
    # 현재 환경의 디폴트 조인트 포지션 복사
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    
    # 강제로 수직 자세 값 덮어쓰기
    joint_pos[:] = target_pose

    # 매우 작은 노이즈 추가 (학습 다양성 및 초기화 안정성)
    noise = (torch.rand_like(joint_pos) - 0.5) * 0.01
    joint_pos += noise

    # 속도는 0으로 초기화
    joint_vel = torch.zeros_like(joint_pos)
    
    # 시뮬레이터에 상태 적용
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
