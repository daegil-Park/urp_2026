# SPDX-License-Identifier: BSD-3-Clause
"""
Observation utilities for Path Tracking Task.
Updated for Contact-Rich Manipulation (Prevent Flailing & Ensure Contact)
"""

from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_mul, quat_conjugate

# [중요] Path Loader 임포트
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


# ------------------------------------------------------
# Global buffers
# ------------------------------------------------------
_EE_HISTORY_BUFFER = None
_EE_HISTORY_LEN = None


# ------------------------------------------------------
# Utility: Contact Sensor Forces
# ------------------------------------------------------
def get_contact_forces(env: ManagerBasedRLEnv, sensor_name="contact_forces"):
    """Z축 힘(수직 항력) 반환"""
    if sensor_name not in env.scene.sensors:
        return torch.zeros((env.num_envs, 1), device=env.device)

    sensor = env.scene.sensors[sensor_name]
    forces_w = sensor.data.net_forces_w
    
    # Z축 힘의 크기 (절댓값)
    force_z = torch.abs(forces_w[..., 2]).unsqueeze(-1)  # (num_envs, 1)
    
    # 너무 큰 값은 클리핑 (학습 안정성)
    force_z = torch.clamp(force_z, 0.0, 50.0)
    return force_z


# ------------------------------------------------------
# Main Observation: Path Tracking (논문 기반 강화)
# ------------------------------------------------------
def path_tracking_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """
    [강화된 관측]
    1. 위치 오차 (Position Error)
    2. 자세 오차 (Orientation Error) - 수직에서 얼마나 벗어났는가?
    3. 속도 정보 (Velocity) - 진동 방지 필수 요소
    4. 힘 정보 (Force)
    """
    device = env.device
    robot = env.scene["robot"]

    # 1. Path Data (목표점 찾기)
    path_data = path_loader.get_path_tensor(device)
    
    # 2. EE State (Pos, Quat, Vel)
    ee_pos = robot.data.body_pos_w[:, -1, :]   # (num_envs, 3)
    ee_quat = robot.data.body_quat_w[:, -1, :] # (num_envs, 4)
    
    # [NEW] 속도 정보 추가 (진동/휘젓기 방지)
    ee_vel_lin = robot.data.body_vel_w[:, -1, :3] # 선속도
    ee_vel_ang = robot.data.body_vel_w[:, -1, 3:] # 각속도

    # 3. Nearest Point Calculation
    # 현재 EE 위치에서 가장 가까운 경로점 찾기
    dists = torch.norm(ee_pos.unsqueeze(1) - path_data.unsqueeze(0), dim=2)
    min_dists, min_indices = torch.min(dists, dim=1)
    target_pos = path_data[min_indices] 

    # 4. Position Error
    pos_error = target_pos - ee_pos

    # 5. [NEW] Orientation Error (수직 정렬 오차)
    # 목표: World -Z 방향 (0, 0, -1)
    # 현재 Tool의 Z축 벡터 계산
    tool_z_local = torch.zeros((env.num_envs, 3), device=device)
    tool_z_local[:, 2] = 1.0
    
    # Quat Rotate (Local -> World)
    # math_utils의 quat_apply가 없으면 직접 구현하거나 isaaclab utils 사용
    # 여기서는 간략화된 로직 사용 (IsaacLab 내부 함수 가정)
    from isaaclab.utils.math import quat_apply
    tool_z_world = quat_apply(ee_quat, tool_z_local)
    
    # 수직(-Z)과의 내적 (1.0에 가까울수록 수직)
    vertical_alignment = tool_z_world[:, 2].unsqueeze(-1) # -1.0(위) ~ 1.0(아래)?? 
    # 주의: World Z는 위가 +Z. Tool Z가 아래(-Z)를 향해야 함.
    # Tool Z(Local +Z)가 World -Z와 같아야 하므로, tool_z_world.z는 -1이어야 함.
    # 따라서 정렬 오차는 tool_z_world.z - (-1) => tool_z_world.z + 1
    
    # 6. Force
    force_z = get_contact_forces(env)

    # 7. 관측 벡터 결합 (총 14차원)
    # - pos_error (3)
    # - vertical_alignment (1) -> 단순 Quat보다 학습이 훨씬 빠름
    # - ee_vel_lin (3) -> 진동 억제용
    # - ee_vel_ang (3) -> 회전 진동 억제용
    # - force_z (1)
    # - ee_pos (3) -> 절대 위치 참고용
    return torch.cat([
        pos_error, 
        vertical_alignment, 
        ee_vel_lin, 
        ee_vel_ang, 
        force_z, 
        ee_pos
    ], dim=-1)


# ------------------------------------------------------
# Observation: History Buffer
# ------------------------------------------------------
def ee_pose_history(env: ManagerBasedRLEnv, history_len: int = 5) -> torch.Tensor:
    global _EE_HISTORY_BUFFER, _EE_HISTORY_LEN
    
    robot = env.scene["robot"]
    # Pos(3) + Quat(4) = 7
    current_pose = torch.cat([
        robot.data.body_pos_w[:, -1, :], 
        robot.data.body_quat_w[:, -1, :]
    ], dim=-1)

    if _EE_HISTORY_BUFFER is None or _EE_HISTORY_BUFFER.shape[0] != env.num_envs:
        _EE_HISTORY_BUFFER = torch.zeros((env.num_envs, history_len, 7), device=env.device)
        _EE_HISTORY_LEN = history_len

    _EE_HISTORY_BUFFER = torch.roll(_EE_HISTORY_BUFFER, shifts=-1, dims=1)
    _EE_HISTORY_BUFFER[:, -1, :] = current_pose

    return _EE_HISTORY_BUFFER.reshape(env.num_envs, -1)
