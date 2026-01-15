# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

# [중요] 방금 만든 로더 임포트 (같은 폴더에 있어야 함)
from . import path_loader

## 히스토리 버퍼
EE_HISTORY_BUFFER = None
EE_HISTORY_LEN = None

def path_tracking_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """ 로봇 EE와 txt 파일 경로 사이의 관계 관측 """
    device = env.device
    num_envs = env.num_envs
    
    # 1. txt 파일 데이터 가져오기
    path_data = path_loader.get_path_tensor(device)

    # 2. 로봇 현재 위치
    robot = env.scene["robot"]
    ee_pos = robot.data.body_pos_w[:, -1, :]  # (num_envs, 3)

    # 3. 가장 가까운 목표점 찾기 (Distance Based)
    # (N, 1, 3) - (1, M, 3) -> 거리 계산
    dists = torch.norm(ee_pos.unsqueeze(1) - path_data.unsqueeze(0), dim=2)
    min_dists, min_indices = torch.min(dists, dim=1)
    
    # 목표 위치
    target_pos = path_data[min_indices] 

    # 4. 오차 계산
    pos_error = target_pos - ee_pos

    # 5. 힘 센서 (있으면 사용, 없으면 0)
    if "contact_forces" in env.scene.sensors:
        sensor_data = env.scene["contact_forces"].data.net_forces_w
        force_z = torch.norm(sensor_data[..., 2], dim=-1, keepdim=True)
        force_z = torch.clamp(force_z, 0.0, 50.0)
    else:
        force_z = torch.zeros((num_envs, 1), device=device)

    # 6. 관측 반환
    return torch.cat([pos_error, force_z, ee_pos], dim=-1)


def ee_pose_history(env: ManagerBasedRLEnv, history_len: int = 5) -> torch.Tensor:
    """ EE 위치 히스토리 """
    global EE_HISTORY_BUFFER, EE_HISTORY_LEN
    
    robot = env.scene["robot"]
    current_pose = torch.cat([
        robot.data.body_pos_w[:, -1, :], 
        robot.data.body_quat_w[:, -1, :]
    ], dim=-1)

    if EE_HISTORY_BUFFER is None or EE_HISTORY_BUFFER.shape[0] != env.num_envs:
        EE_HISTORY_BUFFER = torch.zeros((env.num_envs, history_len, 7), device=env.device)
        EE_HISTORY_LEN = history_len

    EE_HISTORY_BUFFER = torch.roll(EE_HISTORY_BUFFER, shifts=-1, dims=1)
    EE_HISTORY_BUFFER[:, -1, :] = current_pose

    return EE_HISTORY_BUFFER.reshape(env.num_envs, -1)
