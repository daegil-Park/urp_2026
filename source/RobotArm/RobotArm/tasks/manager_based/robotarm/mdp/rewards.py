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
    combine_frame_transforms, quat_error_magnitude
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
    from nrs_fk_core import FKSolver
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver

from RobotArm.robots.ur10e_w_spindle import *

# -----------------------------------------------------------
# Global Logging Variables (Code B 스타일 적용)
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
    # path_manager를 통해 CSV 경로 반환 (path_loader 내부 로직 의존)
    # env에 pm(PathManager)이 인스턴스로 존재한다고 가정
    if hasattr(env, "pm"):
        return env.pm.get_target_pose_from_path(env)
    
    # pm이 없는 경우를 대비한 Fallback (또는 path_loader의 전역 함수 사용)
    # 여기서는 path_loader가 전역적으로 관리된다면 아래와 같이 처리 가능
    # return path_loader.get_target_pose(env)
    return torch.zeros(env.num_envs, 3, device=env.device)

# -----------------------------------------------------------
# Reward Functions
# -----------------------------------------------------------

def track_path_reward(env: ManagerBasedRLEnv, sigma: float = 0.1):
    """
    [경로 추종] 현재 위치가 경로상의 가장 가까운 점과 얼마나 가까운지 평가
    + Code B 스타일의 로깅 및 시각화 트리거 추가
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
    # (num_envs, 1, 3) - (1, num_points, 3) -> (num_envs, num_points, 3)
    dists = torch.norm(current_pos.unsqueeze(1) - path_tensor.unsqueeze(0), dim=2)
    
    # 4. 가장 가까운 거리(Minimum Distance) 추출
    min_dist, _ = torch.min(dists, dim=1)

    # 5. [Logging] 첫 번째 환경(env 0)에 대해서만 기록
    step = int(env.common_step_counter)
    # 기록: (Step, Min_Distance)
    _path_tracking_history.append((step, min_dist[0].item()))

    # 6. [Visualization Trigger] 에피소드 종료 시 그래프 저장
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        # 마지막 스텝에서 저장
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
        # net_forces_w: (num_envs, num_links, 3)
        # 센서 데이터가 비어있지 않은지 확인
        if sensor.data.net_forces_w is not None and sensor.data.net_forces_w.shape[1] > 0:
            force_z = torch.abs(sensor.data.net_forces_w[..., 2])
            # 여러 링크 중 최대 힘 사용
            current_force, _ = torch.max(force_z, dim=-1)
    
    # 2. 오차 계산
    force_error = torch.abs(current_force - target_force)

    # 3. [Logging] 첫 번째 환경(env 0)에 대해서만 기록
    # 기록: (Step, Current_Force, Target_Force)
    step = int(env.common_step_counter)
    _force_control_history.append((step, current_force[0].item(), target_force))

    # 4. 보상 반환 (분모에 epsilon 추가하여 안전성 확보)
    return 1.0 / (1.0 + 0.1 * force_error)


def orientation_align_reward(env: ManagerBasedRLEnv):
    """
    [자세 유지] Tool이 바닥(-Z)을 수직으로 바라보는지 평가
    Code B의 Math utils 활용 가능성 열어둠
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
    """
    [부드러움] Action 값의 크기 억제 (에너지 최소화)
    """
    return -torch.sum(torch.square(env.action_manager.action), dim=-1)


def out_of_bounds_penalty(env: ManagerBasedRLEnv):
    """
    [이탈 방지] 작업 영역 벗어나면 벌점
    """
    ee_pos = get_ee_pose(env)[:, :3]
    
    # 작업 영역 설정 (필요시 env.cfg에서 가져오도록 수정 가능)
    wp_pos_x, wp_pos_y = 0.5, 0.0
    wp_size_x, wp_size_y = 0.6, 0.6 

    is_out_x = (ee_pos[:, 0] < (wp_pos_x - wp_size_x)) | (ee_pos[:, 0] > (wp_pos_x + wp_size_x))
    is_out_y = (ee_pos[:, 1] < (wp_pos_y - wp_size_y)) | (ee_pos[:, 1] > (wp_pos_y + wp_size_y))
    is_out_z = (ee_pos[:, 2] < 0.0) | (ee_pos[:, 2] > 0.6) # 상한선 약간 여유

    is_out = (is_out_x | is_out_y | is_out_z).float()
    
    return -1.0 * is_out


# -----------------------------------------------------------
# Visualization Logic (From Code B)
# -----------------------------------------------------------
def save_episode_plots(step: int):
    """에피소드 종료 시 데이터를 그래프로 저장"""
    global _path_tracking_history, _force_control_history, _episode_counter
    
    # 저장 경로 설정
    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Path Tracking Plot
    if _path_tracking_history:
        steps, dists = zip(*_path_tracking_history)
        
        plt.figure(figsize=(10, 5))
        plt.plot(steps, dists, label="Distance to Path", color="blue")
        plt.title(f"Path Tracking Error (Ep {_episode_counter + 1})")
        plt.xlabel("Step")
        plt.ylabel("Distance [m]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"path_error_ep{_episode_counter + 1}.png"))
        plt.close()

    # 2. Force Control Plot
    if _force_control_history:
        steps, currents, targets = zip(*_force_control_history)
        
        plt.figure(figsize=(10, 5))
        plt.plot(steps, currents, label="Current Force", color="red")
        plt.plot(steps, targets, "--", label="Target Force", color="green")
        plt.title(f"Force Control (Ep {_episode_counter + 1})")
        plt.xlabel("Step")
        plt.ylabel("Force [N]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"force_control_ep{_episode_counter + 1}.png"))
        plt.close()

    # 데이터 초기화 및 카운터 증가
    _path_tracking_history.clear()
    _force_control_history.clear()
    _episode_counter += 1
    print(f"[{step}] Episode {_episode_counter} plots saved to {save_dir}")


def check_coverage_success(env: ManagerBasedRLEnv):
    """종료 조건: 기본적으로 False (타임아웃까지 수행)"""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def manual_termination(env: ManagerBasedRLEnv):
    """
    'K' 키를 누르면 강제 종료 (리셋)
    headless 모드에서 carb가 없을 경우 안전하게 패스
    """
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
