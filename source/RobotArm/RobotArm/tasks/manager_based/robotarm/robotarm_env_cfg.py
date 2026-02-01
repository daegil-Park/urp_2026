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
# [중요] Ideal Joint Pose (수직 작업 자세)
# -----------------------------------------------------------
# robotarm_env_cfg.py의 DEVICE_READY_STATE와 값이 동일해야 합니다.
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
IDEAL_JOINT_POSE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

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

    # Visualization
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots(step)

    return torch.exp(-torch.square(min_dist) / (sigma ** 2))


def orientation_align_reward(env: ManagerBasedRLEnv):
    """[수직 자세 강제]"""
    ee_pose = get_ee_pose(env)
    ee_quat = ee_pose[:, 3:]

    tool_z_local = torch.zeros((env.num_envs, 3), device=env.device)
    tool_z_local[:, 2] = 1.0 
    tool_z_world = quat_apply(ee_quat, tool_z_local)
    
    target_dir = torch.zeros_like(tool_z_world)
    target_dir[:, 2] = -1.0 

    dot_prod = torch.sum(tool_z_world * target_dir, dim=-1)
    error = 1.0 - torch.clamp(dot_prod, min=-1.0, max=1.0)
    
    # 강력한 페널티 (Scale 30.0)
    return torch.exp(-error * 30.0)


def joint_deviation_reward(env: ManagerBasedRLEnv):
    """[자세 유지] IDEAL_JOINT_POSE에서 벗어나지 않도록"""
    robot = env.scene["robot"]
    current_joints = robot.data.joint_pos # (num_envs, 6)
    
    target_joints = torch.tensor(IDEAL_JOINT_POSE, device=env.device).unsqueeze(0)
    
    diff = torch.norm(current_joints - target_joints, dim=-1)
    return torch.exp(-diff * 0.5)


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

    step = int(env.common_step_counter)
    if env.num_envs > 0:
        _force_control_history.append((step, current_force[0].item(), target_force))

    return 1.0 / (1.0 + 0.1 * force_error)

def action_smoothness_penalty(env: ManagerBasedRLEnv):
    return -torch.sum(torch.square(env.action_manager.action), dim=-1)

def out_of_bounds_penalty(env: ManagerBasedRLEnv):
    ee_pos = get_ee_pose(env)[:, :3]
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
    [일치화 완료] robotarm_env_cfg.py의 초기 자세와 동일하게 리셋
    """
    robot = env.scene["robot"]
    
    target_pose = torch.tensor(IDEAL_JOINT_POSE, device=env.device)
    
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    
    # 디폴트가 이미 IDEAL_JOINT_POSE로 설정되어 있으므로, 
    # 그냥 default_joint_pos를 써도 되지만, 확실히 하기 위해 값을 덮어씁니다.
    joint_pos[:] = target_pose

    # 아주 미세한 노이즈 (학습 안정성)
    noise = (torch.rand_like(joint_pos) - 0.5) * 0.02
    joint_pos += noise

    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
