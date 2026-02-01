# SPDX-License-Identifier: BSD-3-Clause
"""
Observation utilities for Path Tracking Task.
- Integrated with path_loader (TXT based waypoint generation)
- Includes EE pose (Sim-based or FK-based), contact, and history buffers
- optimized for Isaac Lab tensor API
"""

from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

# [중요] Path Loader 임포트
from . import path_loader

# ✅ 조건부 import (FK 모듈, 필요시 사용)
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


# ------------------------------------------------------
# Global buffers
# ------------------------------------------------------
_EE_HISTORY_BUFFER = None
_EE_HISTORY_LEN = None


# ------------------------------------------------------
# Utility: EE Pose (Performance Optimized)
# ------------------------------------------------------
def get_ee_pose(env: ManagerBasedRLEnv, asset_name: str = "robot", use_fk: bool = False):
    """
    Returns end-effector pose (x, y, z, roll, pitch, yaw)
    
    Args:
        use_fk (bool): 
            - False (Default): Use Simulator Ground Truth (Fast, for RL training)
            - True: Use FKSolver (Slower, for validation or hardware transfer)
    """
    robot = env.scene[asset_name]
    
    if use_fk and FKSolver is not None:
        # [Option A] FK Solver 사용
        q = robot.data.joint_pos[:, :6]
        num_envs = q.shape[0]
        fk_solver = FKSolver(tool_z=0.239, use_degrees=False)
        ee_pose_list = []
        
        for i in range(num_envs):
            q_np = q[i].cpu().numpy().astype(float)
            ok, pose = fk_solver.compute(q_np, as_degrees=False)
            if not ok:
                ee_pose_list.append([0.0]*6)
            else:
                ee_pose_list.append([pose.x, pose.y, pose.z, pose.r, pose.p, pose.yaw])
        return torch.tensor(ee_pose_list, dtype=torch.float32, device=env.device)
    
    else:
        # [Option B] Sim Ground Truth 사용 (Fast, Recommended for RL)
        pos = robot.data.body_pos_w[:, -1, :]  # (num_envs, 3)
        quat = robot.data.body_quat_w[:, -1, :] # (num_envs, 4)
        r, p, y = euler_xyz_from_quat(quat)
        
        # Stack into (num_envs, 6)
        return torch.cat([pos, r.unsqueeze(1), p.unsqueeze(1), y.unsqueeze(1)], dim=-1)


# ------------------------------------------------------
# Utility: Contact Sensor Forces
# ------------------------------------------------------
def get_contact_forces(env: ManagerBasedRLEnv, sensor_name="contact_forces"):
    """
    Returns mean contact wrench or Z-force magnitude.
    Handles missing sensors gracefully.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros((env.num_envs, 1), device=env.device)

    sensor = env.scene.sensors[sensor_name]
    # net_forces_w: (num_envs, num_links, 3)
    forces_w = sensor.data.net_forces_w
    
    # Z축 힘의 크기(Norm)
    force_z = torch.norm(forces_w[..., 2], dim=-1, keepdim=True)  # (num_envs, 1)
    
    # Clipping (Safety)
    force_z = torch.clamp(force_z, 0.0, 50.0)
    
    return force_z


# ------------------------------------------------------
# Main Observation: Path Tracking (핵심 수정됨)
# ------------------------------------------------------
def path_tracking_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """
    로봇 EE와 txt 파일 경로(path_loader) 사이의 관계를 관측합니다.
    [변경점] 반환값에 ee_quat(현재 회전)을 추가하여 로봇이 자신의 기울기를 알 수 있게 함.
    """
    device = env.device
    
    # 1. txt 파일 데이터 가져오기 (via path_loader)
    path_data = path_loader.get_path_tensor(device)

    # 2. 로봇 현재 위치 및 회전 (Sim Ground Truth)
    robot = env.scene["robot"]
    ee_pos = robot.data.body_pos_w[:, -1, :]   # (num_envs, 3)
    ee_quat = robot.data.body_quat_w[:, -1, :] # (num_envs, 4) -> [NEW] 추가됨

    # 3. 가장 가까운 목표점 찾기 (Distance Based)
    dists = torch.norm(ee_pos.unsqueeze(1) - path_data.unsqueeze(0), dim=2)
    min_dists, min_indices = torch.min(dists, dim=1)
    
    # 해당 인덱스의 목표 위치 가져오기
    target_pos = path_data[min_indices]  # (num_envs, 3)

    # 4. 오차 계산 (Vector)
    pos_error = target_pos - ee_pos

    # 5. 힘 센서 데이터
    force_z = get_contact_forces(env)

    # 6. 관측 반환 (Pos Error + Force + Current Pos + Current Rot)
    # 기존: 3 + 1 + 3 = 7
    # 변경: 3(error) + 1(force) + 3(pos) + 4(quat) = 11 차원
    return torch.cat([pos_error, force_z, ee_pos, ee_quat], dim=-1)


# ------------------------------------------------------
# Observation: History Buffer
# ------------------------------------------------------
def ee_pose_history(env: ManagerBasedRLEnv, history_len: int = 5) -> torch.Tensor:
    """EE 위치 및 회전 히스토리 버퍼 관리"""
    global _EE_HISTORY_BUFFER, _EE_HISTORY_LEN
    
    # 현재 Pose 가져오기 (Pos + Quat = 7 dim)
    robot = env.scene["robot"]
    current_pose = torch.cat([
        robot.data.body_pos_w[:, -1, :], 
        robot.data.body_quat_w[:, -1, :]
    ], dim=-1)

    # 버퍼 초기화
    if _EE_HISTORY_BUFFER is None or _EE_HISTORY_BUFFER.shape[0] != env.num_envs:
        _EE_HISTORY_BUFFER = torch.zeros((env.num_envs, history_len, 7), device=env.device)
        _EE_HISTORY_LEN = history_len

    # 버퍼 업데이트 (Shift & Insert)
    _EE_HISTORY_BUFFER = torch.roll(_EE_HISTORY_BUFFER, shifts=-1, dims=1)
    _EE_HISTORY_BUFFER[:, -1, :] = current_pose

    # Flatten하여 반환 (num_envs, history_len * 7)
    return _EE_HISTORY_BUFFER.reshape(env.num_envs, -1)
