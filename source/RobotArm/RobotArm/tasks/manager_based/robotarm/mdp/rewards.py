# 파일 경로: RobotArm/tasks/manager_based/robotarm/mdp/rewards.py
from __future__ import annotations

import torch
import math
import os
import numpy as np
from typing import TYPE_CHECKING

# [시각화 관련]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab.envs import ManagerBasedRLEnv

# [입력 관련]
try:
    import carb.input
except ImportError:
    carb = None

# [데이터 로더]
from . import path_loader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -----------------------------------------------------------
# Global Logging Variables
# -----------------------------------------------------------
_path_tracking_history = []
_force_control_history = []
_episode_counter = 0

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def get_ee_pose(env: ManagerBasedRLEnv, asset_name: str = "robot"):
    robot = env.scene[asset_name]
    pos = robot.data.body_pos_w[:, -1, :]
    quat = robot.data.body_quat_w[:, -1, :]
    return torch.cat([pos, quat], dim=-1)

# -----------------------------------------------------------
# Reward Functions
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

    # Logging
    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _path_tracking_history.append((step, min_dist[0].item()))

    # Plot Save
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

    # Logging
    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _force_control_history.append((step, current_force[0].item(), target_force))

    return 1.0 / (1.0 + 0.1 * force_error)


def orientation_align_reward(env: ManagerBasedRLEnv):
    """[자세 유지] Tool이 바닥(-Z)을 바라보는지"""
    ee_pose = get_ee_pose(env)
    ee_quat = ee_pose[:, 3:]

    tool_z_local = torch.zeros((env.num_envs, 3), device=env.device)
    tool_z_local[:, 2] = 1.0 
    
    tool_z_world = quat_apply(ee_quat, tool_z_local)

    target_dir = torch.zeros_like(tool_z_world)
    target_dir[:, 2] = -1.0 

    dot_prod = torch.sum(tool_z_world * target_dir, dim=-1)
    
    return torch.clamp(dot_prod, min=-1.0, max=1.0)


def action_smoothness_penalty(env: ManagerBasedRLEnv):
    """[부드러움]"""
    return -torch.sum(torch.square(env.action_manager.action), dim=-1)


def out_of_bounds_penalty(env: ManagerBasedRLEnv):
    """[이탈 방지]"""
    ee_pos = get_ee_pose(env)[:, :3]
    
    wp_pos_x, wp_pos_y = 0.5, 0.0
    wp_size_x, wp_size_y = 0.6, 0.6 

    is_out_x = (ee_pos[:, 0] < (wp_pos_x - wp_size_x)) | (ee_pos[:, 0] > (wp_pos_x + wp_size_x))
    is_out_y = (ee_pos[:, 1] < (wp_pos_y - wp_size_y)) | (ee_pos[:, 1] > (wp_pos_y + wp_size_y))
    is_out_z = (ee_pos[:, 2] < 0.0) | (ee_pos[:, 2] > 0.8)

    is_out = (is_out_x | is_out_y | is_out_z).float()
    
    return -1.0 * is_out

def pen_table_collision(env: ManagerBasedRLEnv, threshold: float = 0.0):
    """[충돌 방지] threshold보다 낮으면 강력한 벌점"""
    ee_pos = get_ee_pose(env)[:, :3]
    
    is_under = (ee_pos[:, 2] < (threshold - 0.01)).float()
    penetration = (threshold - ee_pos[:, 2]).clamp(min=0.0)
    
    return is_under * (-1.0 - penetration * 10.0)

def rew_surface_tracking(env: ManagerBasedRLEnv, target_height: float = 0.0):
    """[표면 밀착]"""
    ee_pos = get_ee_pose(env)[:, :3]
    z_error = torch.abs(ee_pos[:, 2] - target_height)
    return torch.exp(-z_error / 0.02)


# -----------------------------------------------------------
# [NEW] 강력한 초기화 함수 (Forced Reset)
# -----------------------------------------------------------
def reset_robot_to_cobra(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    로봇을 무조건 Cobra Pose(작업 자세)로 강제 설정합니다.
    기본 reset 함수가 USD의 default state(차렷 자세)를 불러오는 문제를 해결합니다.
    """
    # 1. 로봇 Asset 가져오기
    robot = env.scene["robot"]

    # 2. Cobra Pose 정의 (6 DOF)
    # [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
    cobra_pose = torch.tensor([0.0, -2.0, 2.0, -1.57, -1.57, 0.0], device=env.device)
    
    # 3. 현재 관절 위치 가져오기
    joint_pos = robot.data.default_joint_pos[env_ids].clone()

    # 4. 모든 관절 값을 Cobra Pose로 덮어쓰기
    # (Broadcasting: env_ids 개수만큼 복사)
    joint_pos[:] = cobra_pose

    # 5. 아주 미세한 랜덤 노이즈 추가 (학습 다양성)
    noise = (torch.rand_like(joint_pos) - 0.5) * 0.02
    joint_pos += noise

    # 6. 속도는 0으로 초기화
    joint_vel = torch.zeros_like(joint_pos)

    # 7. 로봇 상태 강제 설정
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    
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
