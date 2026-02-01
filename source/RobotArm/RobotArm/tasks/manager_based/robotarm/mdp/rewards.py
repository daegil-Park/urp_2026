# 파일 경로: RobotArm/tasks/manager_based/robotarm/mdp/rewards.py
from __future__ import annotations

import torch
import math
import os
import csv
import numpy as np
from typing import TYPE_CHECKING

# [복구] 시각화 관련 라이브러리
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

# [복구] 입력 인터페이스
try:
    import carb.input
except ImportError:
    carb = None

from pxr import UsdGeom

# [복구] 데이터 로더 임포트
from . import path_loader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
# [복구] FK Solver (이게 없어서 에러 났을 겁니다)
if "nrs_fk_core" not in sys.modules:
    try:
        from nrs_fk_core import FKSolver
    except ImportError:
        FKSolver = None
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver

# [복구] 로봇 정의
try:
    from RobotArm.robots.ur10e_w_spindle import *
except ImportError:
    pass

# -----------------------------------------------------------
# Global Logging Variables (기존 유지)
# -----------------------------------------------------------
_path_tracking_history = []
_force_control_history = []
_episode_counter = 0

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return diff

def get_ee_pose(env: ManagerBasedRLEnv, asset_name: str = "robot"):
    """Sim에서 제공하는 정확한 물리적 위치와 쿼터니언을 반환합니다."""
    robot = env.scene[asset_name]
    pos = robot.data.body_pos_w[:, -1, :]
    quat = robot.data.body_quat_w[:, -1, :]
    return torch.cat([pos, quat], dim=-1)

def _get_generated_target(env):
    if hasattr(env, "pm"):
        return env.pm.get_target_pose_from_path(env)
    return torch.zeros(env.num_envs, 3, device=env.device)

# -----------------------------------------------------------
# Reward Functions (삭제된 것 없이 전부 복구)
# -----------------------------------------------------------

def track_path_reward(env: ManagerBasedRLEnv, sigma: float = 0.1):
    """[경로 추종]"""
    global _path_tracking_history, _episode_counter

    path_tensor = path_loader.get_path_tensor(env.device)
    if path_tensor.device != env.device:
        path_tensor = path_tensor.to(env.device)

    ee_pose = get_ee_pose(env)
    current_pos = ee_pose[:, :3]

    dists = torch.norm(current_pos.unsqueeze(1) - path_tensor.unsqueeze(0), dim=2)
    min_dist, _ = torch.min(dists, dim=1)

    # [Logging]
    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _path_tracking_history.append((step, min_dist[0].item()))

    # [Visualization Trigger] 에피소드 종료 시 그래프 저장
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots(step)

    return torch.exp(-torch.square(min_dist) / (sigma ** 2))


def force_control_reward(env: ManagerBasedRLEnv, target_force: float = 10.0):
    """[힘 제어]"""
    global _force_control_history

    current_force = torch.zeros(env.num_envs, device=env.device)

    if "contact_forces" in env.scene.sensors:
        sensor = env.scene["contact_forces"]
        if sensor.data.net_forces_w is not None and sensor.data.net_forces_w.shape[1] > 0:
            force_z = torch.abs(sensor.data.net_forces_w[..., 2])
            current_force, _ = torch.max(force_z, dim=-1)
    
    force_error = torch.abs(current_force - target_force)

    # [Logging]
    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _force_control_history.append((step, current_force[0].item(), target_force))

    return 1.0 / (1.0 + 0.1 * force_error)


def orientation_align_reward(env: ManagerBasedRLEnv):
    """[자세 유지]"""
    ee_pose = get_ee_pose(env)
    ee_quat = ee_pose[:, 3:] 

    tool_z_local = torch.zeros((env.num_envs, 3), device=env.device)
    tool_z_local[:, 2] = 1.0 
    
    tool_z_world = quat_apply(ee_quat, tool_z_local)
    target_dir = torch.zeros_like(tool_z_world)
    target_dir[:, 2] = -1.0 

    dot_prod = torch.sum(tool_z_world * target_dir, dim=-1)
    return torch.clamp(dot_prod, min=0.0)
    

def action_smoothness_penalty(env: ManagerBasedRLEnv):
    return -torch.sum(torch.square(env.action_manager.action), dim=-1)


def out_of_bounds_penalty(env: ManagerBasedRLEnv):
    ee_pos = get_ee_pose(env)[:, :3]
    wp_pos_x, wp_pos_y = 0.5, 0.0
    wp_size_x, wp_size_y = 0.6, 0.6 

    is_out_x = (ee_pos[:, 0] < (wp_pos_x - wp_size_x)) | (ee_pos[:, 0] > (wp_pos_x + wp_size_x))
    is_out_y = (ee_pos[:, 1] < (wp_pos_y - wp_size_y)) | (ee_pos[:, 1] > (wp_pos_y + wp_size_y))
    is_out_z = (ee_pos[:, 2] < 0.0) | (ee_pos[:, 2] > 0.8) 

    is_out = (is_out_x | is_out_y | is_out_z).float()
    return -1.0 * is_out

# -----------------------------------------------------------
# [NEW] 물리 충돌 방지 및 표면 밀착 (새로 추가)
# -----------------------------------------------------------

def pen_table_collision(env: ManagerBasedRLEnv, threshold: float = 0.0):
    """테이블 뚫으면 강력한 벌점"""
    ee_pos = get_ee_pose(env)[:, :3]
    is_under = (ee_pos[:, 2] < (threshold - 0.01)).float()
    penetration = (threshold - ee_pos[:, 2]).clamp(min=0.0)
    return is_under * (-1.0 - penetration * 10.0)

def rew_surface_tracking(env: ManagerBasedRLEnv, target_height: float = 0.0):
    """표면 높이 유지"""
    ee_pos = get_ee_pose(env)[:, :3]
    z_error = torch.abs(ee_pos[:, 2] - target_height)
    return torch.exp(-z_error / 0.02)


# -----------------------------------------------------------
# Visualization Logic (이 부분 다 날려서 죄송합니다. 복구했습니다.)
# -----------------------------------------------------------
def save_episode_plots(step: int):
    """에피소드 종료 시 그래프 저장"""
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

# -----------------------------------------------------------
# [NEW - 여기가 핵심] 강제 리셋 함수
# -----------------------------------------------------------
def reset_robot_to_cobra(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    로봇이 태어날 때(Reset) USD 기본 자세(차렷)가 아니라,
    무조건 안전한 'Cobra Pose'로 시작하도록 강제합니다.
    """
    robot = env.scene["robot"]

    # [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
    # 팔을 뒤로 젖히고 앞으로 숙인 자세 (안전 자세)
    cobra_pose = torch.tensor([0.0, -2.0, 2.0, -1.57, -1.57, 0.0], device=env.device)
    
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:] = cobra_pose

    # 약간의 랜덤성 (학습 다양성)
    noise = (torch.rand_like(joint_pos) - 0.5) * 0.02
    joint_pos += noise

    joint_vel = torch.zeros_like(joint_pos)

    # 시뮬레이션에 강제 주입
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
