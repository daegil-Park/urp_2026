# 파일 경로: RobotArm/tasks/manager_based/robotarm/mdp/rewards.py
from __future__ import annotations

import torch
import math
import os
import csv
import numpy as np
from typing import TYPE_CHECKING

# [시각화 관련] Headless 모드 지원을 위해 Agg 백엔드 설정
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    matrix_from_euler, quat_apply, quat_mul, 
    quat_from_euler_xyz, euler_xyz_from_quat, 
    combine_frame_transforms, quat_error_magnitude, quat_rotate
)
from isaaclab.envs import ManagerBasedRLEnv

# [입력 관련] 예외 처리 추가
try:
    import carb.input
except ImportError:
    carb = None

from pxr import UsdGeom

# [중요] 데이터 로더 임포트 (경로 유지)
from . import path_loader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
if "nrs_fk_core" not in sys.modules:
    try:
        from nrs_fk_core import FKSolver
    except ImportError:
        FKSolver = None
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver

# 로봇 정의가 있다면 유지 (없으면 패스)
try:
    from RobotArm.robots.ur10e_w_spindle import *
except ImportError:
    pass

# -----------------------------------------------------------
# Global Logging Variables (기존 코드 유지)
# -----------------------------------------------------------
_path_tracking_history = []
_force_control_history = []
_episode_counter = 0

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 각도(radian)의 차이를 [-pi, pi] 범위로 계산합니다."""
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return diff

def get_ee_pose(env: ManagerBasedRLEnv, asset_name: str = "robot"):
    """Sim에서 제공하는 정확한 물리적 위치와 쿼터니언을 반환합니다."""
    robot = env.scene[asset_name]
    # body_pos_w: (num_envs, num_bodies, 3) -> 마지막 바디(Tool) 선택
    pos = robot.data.body_pos_w[:, -1, :]
    quat = robot.data.body_quat_w[:, -1, :]
    return torch.cat([pos, quat], dim=-1)

# -----------------------------------------------------------
# Helper Functions for Target
# -----------------------------------------------------------
def _get_generated_target(env):
    """observation_1.py와 동일한 로직을 유지"""
    if hasattr(env, "pm"):
        return env.pm.get_target_pose_from_path(env)
    return torch.zeros(env.num_envs, 3, device=env.device)

# -----------------------------------------------------------
# Reward Functions (Original + New Solutions)
# -----------------------------------------------------------

def track_path_reward(env: ManagerBasedRLEnv, sigma: float = 0.1):
    """
    [경로 추종] 현재 위치가 경로상의 가장 가까운 점과 얼마나 가까운지 평가
    """
    global _path_tracking_history, _episode_counter

    # 1. Path Tensor 가져오기 및 디바이스 동기화
    path_tensor = path_loader.get_path_tensor(env.device)
    if path_tensor.device != env.device:
        path_tensor = path_tensor.to(env.device)

    # 2. 현재 로봇 손끝 위치
    ee_pose = get_ee_pose(env)
    current_pos = ee_pose[:, :3] # (num_envs, 3)

    # 3. 경로상의 점들과의 거리 계산 (Broadcasting)
    dists = torch.norm(current_pos.unsqueeze(1) - path_tensor.unsqueeze(0), dim=2)
    
    # 4. 가장 가까운 거리(Minimum Distance) 추출
    min_dist, _ = torch.min(dists, dim=1)

    # 5. [Logging] 첫 번째 환경(env 0)에 대해서만 기록
    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _path_tracking_history.append((step, min_dist[0].item()))

    # 6. [Visualization Trigger] 에피소드 종료 시 그래프 저장
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots(step)

    # 7. 가우시안 커널 보상
    return torch.exp(-torch.square(min_dist) / (sigma ** 2))


def force_control_reward(env: ManagerBasedRLEnv, target_force: float = 10.0):
    """
    [힘 제어] Z축 접촉 힘을 목표값 유지 + 로깅 추가
    """
    global _force_control_history

    current_force = torch.zeros(env.num_envs, device=env.device)

    # 1. 센서 데이터 가져오기
    if "contact_forces" in env.scene.sensors:
        sensor = env.scene["contact_forces"]
        if sensor.data.net_forces_w is not None and sensor.data.net_forces_w.shape[1] > 0:
            force_z = torch.abs(sensor.data.net_forces_w[..., 2])
            current_force, _ = torch.max(force_z, dim=-1)
    
    # 2. 오차 계산
    force_error = torch.abs(current_force - target_force)

    # 3. [Logging]
    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _force_control_history.append((step, current_force[0].item(), target_force))

    # 4. 보상 반환
    return 1.0 / (1.0 + 0.1 * force_error)


def orientation_align_reward(env: ManagerBasedRLEnv):
    """
    [자세 유지 - 기존 버전] Tool이 바닥(-Z)을 수직으로 바라보는지 평가
    """
    ee_pose = get_ee_pose(env)
    ee_quat = ee_pose[:, 3:] # (qx, qy, qz, qw)

    # 1. 로봇 Tool의 Z축 벡터 (Local 0,0,1)
    tool_z_local = torch.zeros((env.num_envs, 3), device=env.device)
    tool_z_local[:, 2] = 1.0 
    
    # 2. 쿼터니언 회전 적용 -> World 좌표계 벡터
    tool_z_world = quat_apply(ee_quat, tool_z_local)

    # 3. 목표 벡터 (World -Z: 0, 0, -1)
    target_dir = torch.zeros_like(tool_z_world)
    target_dir[:, 2] = -1.0 

    # 4. 내적 (Dot Product)
    dot_prod = torch.sum(tool_z_world * target_dir, dim=-1)
    
    return torch.clamp(dot_prod, min=0.0)
    

def action_smoothness_penalty(env: ManagerBasedRLEnv):
    """[부드러움] Action 값의 크기 억제"""
    return -torch.sum(torch.square(env.action_manager.action), dim=-1)


def out_of_bounds_penalty(env: ManagerBasedRLEnv):
    """[이탈 방지] 작업 영역 벗어나면 벌점"""
    ee_pos = get_ee_pose(env)[:, :3]
    
    wp_pos_x, wp_pos_y = 0.5, 0.0
    wp_size_x, wp_size_y = 0.6, 0.6 

    is_out_x = (ee_pos[:, 0] < (wp_pos_x - wp_size_x)) | (ee_pos[:, 0] > (wp_pos_x + wp_size_x))
    is_out_y = (ee_pos[:, 1] < (wp_pos_y - wp_size_y)) | (ee_pos[:, 1] > (wp_pos_y + wp_size_y))
    is_out_z = (ee_pos[:, 2] < 0.0) | (ee_pos[:, 2] > 0.8) # 상한선 여유

    is_out = (is_out_x | is_out_y | is_out_z).float()
    
    return -1.0 * is_out

# -----------------------------------------------------------
# [NEW] 솔루션 추가 함수들 (물리적 제약 강화)
# -----------------------------------------------------------

def pen_table_collision(env: ManagerBasedRLEnv, threshold: float = 0.0):
    """
    [물리 충돌 방지] 작업물 높이(threshold)보다 낮게 내려가면(뚫으면) 강력한 페널티.
    """
    ee_pos = get_ee_pose(env)[:, :3]
    
    # threshold보다 낮으면 1 (충돌), 아니면 0
    # 안전 마진 1cm (0.01) 고려
    is_under = (ee_pos[:, 2] < (threshold - 0.01)).float()
    
    # 뚫고 들어간 깊이
    penetration = (threshold - ee_pos[:, 2]).clamp(min=0.0)
    
    # 충돌 시 기본 벌점(-1.0) + 깊이에 비례한 벌점(-10.0 * depth)
    return is_under * (-1.0 - penetration * 10.0)

def rew_surface_tracking(env: ManagerBasedRLEnv, target_height: float = 0.0):
    """
    [표면 밀착] 공중에서 휘적거리지 않고 표면 높이(Target Height) 근처를 유지하도록 유도.
    """
    ee_pos = get_ee_pose(env)[:, :3]
    
    # Z축 오차 절대값
    z_error = torch.abs(ee_pos[:, 2] - target_height)
    
    # 2cm 이내면 보상 (Sharp Gaussian)
    return torch.exp(-z_error / 0.02)


# -----------------------------------------------------------
# Visualization Logic (기존 코드 유지)
# -----------------------------------------------------------
def save_episode_plots(step: int):
    """에피소드 종료 시 데이터를 그래프로 저장"""
    global _path_tracking_history, _force_control_history, _episode_counter
    
    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    os.makedirs(save_dir, exist_ok=True)

    if _path_tracking_history:
        steps, dists = zip(*_path_tracking_history)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, dists, label="Distance to Path", color="blue")
        plt.title(f"Path Tracking Error (Ep {_episode_counter + 1})")
        plt.xlabel("Step"); plt.ylabel("Distance [m]")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"path_error_ep{_episode_counter + 1}.png"))
        plt.close()

    if _force_control_history:
        steps, currents, targets = zip(*_force_control_history)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, currents, label="Current Force", color="red")
        plt.plot(steps, targets, "--", label="Target Force", color="green")
        plt.title(f"Force Control (Ep {_episode_counter + 1})")
        plt.xlabel("Step"); plt.ylabel("Force [N]")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"force_control_ep{_episode_counter + 1}.png"))
        plt.close()

    _path_tracking_history.clear()
    _force_control_history.clear()
    _episode_counter += 1
    print(f"[{step}] Episode {_episode_counter} plots saved to {save_dir}")


def check_coverage_success(env: ManagerBasedRLEnv):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def manual_termination(env: ManagerBasedRLEnv):
    if carb is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    try:
        input_i = carb.input.acquire_input_interface()
        keyboard = input_i.get_keyboard()
        if input_i.get_keyboard_value(keyboard, carb.input.KeyboardInput.K) > 0.5:
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    except Exception:
        pass
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
