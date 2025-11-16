# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from RobotArm.tasks.manager_based.robotarm.mdp.rewards import get_workpiece_size

import sys
if "nrs_fk_core" not in sys.modules:
    from nrs_fk_core import FKSolver
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver


fk_solver = FKSolver(tool_z=0.239, use_degrees=False)

# 전역 버퍼 (env 별 히스토리 저장용)
EE_HISTORY_BUFFER = None
EE_HISTORY_LEN = None


def grid_mask_state_obs(env, grid_mask_history_len=4):
    """
    Workpiece의 Grid Mask 상태 관찰
    """
    workpiece = env.scene["workpiece"]
    try:
        wp_size_x, wp_size_y = get_workpiece_size(workpiece)
    except Exception as e:
        # USD Prim 경로가 아직 불안정하거나 Manager 초기화 과정에서 Workpiece Asset이 완전히 로드되지 않았을 수 있습니다.
        # 이 경우, 하드코딩된 기본값 (0.5)을 사용합니다.
        wp_size_x, wp_size_y = 0.5, 0.5
        print(f"Workpiece size dynamic read failed in obs: {e}. Using default.") # 디버깅용

    GRID_SIZE = 0.02
    grid_x_num = int(wp_size_x / GRID_SIZE) # 동적으로 계산
    grid_y_num = int(wp_size_y / GRID_SIZE)

    # 2. env.grid_mask가 정의되지 않았다면 초기화
    if not hasattr(env, "grid_mask"):
        # 초기화: 모든 환경에 대해 Falses로 초기화
        # env.grid_mask = torch.zeros((env.num_envs, grid_x_num, grid_y_num), dtype=torch.bool, device=env.device)
        total_dim = grid_x_num * grid_y_num * grid_mask_history_len

        # Manager가 차원만 알 수 있도록 더미 텐서 반환
        return torch.zeros((env.num_envs, total_dim), device=env.device)
    
    current_mask_float = env.grid_mask.float()
    is_reset = (env.episode_length_buf == 0).any()
    
    if not hasattr(env, "_grid_mask_history") or is_reset:
        # 히스토리 텐서의 shape: (num_envs, history_len, grid_x_num, grid_y_num)
        history_shape = (env.num_envs, grid_mask_history_len) + current_mask_float.shape[1:]
        # 초기화: history_len 만큼 현재 Grid Mask로 채우거나 (리셋이 아닌 경우), 0으로 채웁니다.
        if is_reset and hasattr(env, "_grid_mask_history"):
             # 에피소드 리셋 시 0으로 초기화
            env._grid_mask_history = torch.zeros(history_shape, dtype=torch.float, device=env.device)
        else:
            # 환경 시작 시 초기화
            env._grid_mask_history = torch.stack([current_mask_float] * grid_mask_history_len, dim=1)
        
    # 2. 새로운 Grid Mask 상태를 히스토리 큐에 추가 (FIFO)
    new_mask_float_unsqueeze = current_mask_float.unsqueeze(1) 
    
    # 히스토리 업데이트: 가장 오래된 데이터 제거, 최신 데이터 추가
    env._grid_mask_history = torch.cat(
        [env._grid_mask_history[:, 1:], new_mask_float_unsqueeze], dim=1
    )

    # 4. 관찰 벡터 형태로 평탄화 (Flatten)
    # shape: (num_envs, history_len * grid_x_num * grid_y_num)
    obs_vector = env._grid_mask_history.flatten(start_dim=1)
    
    return obs_vector



def ee_pose_history(env, asset_cfg: SceneEntityCfg, history_len: int = 5) -> torch.Tensor:
    """Return end-effector pose history for the given asset.

    Args:
        env: ManagerBasedRLEnv (현재 환경 인스턴스)
        asset_cfg: SceneEntityCfg - 로봇 및 엔드이펙터 body 이름 포함
        history_len: 저장할 히스토리 길이 (기본값 5 step)

    Returns:
        torch.Tensor: shape = [num_envs, history_len * 6]
                      (각 step마다 [x, y, z, roll, pitch, yaw])
    """
    global EE_HISTORY_BUFFER, EE_HISTORY_LEN

    asset: RigidObject = env.scene[asset_cfg.name]
    ee_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # (num_envs, 3)
    ee_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (num_envs, 4)

    # quaternion → roll, pitch, yaw 변환
    roll, pitch, yaw = quat_to_euler(ee_quat)
    ee_pose = torch.cat([ee_pos, torch.stack([roll, pitch, yaw], dim=-1)], dim=-1)

    # 초기화
    if EE_HISTORY_BUFFER is None or EE_HISTORY_LEN != history_len:
        num_envs = ee_pose.shape[0]
        EE_HISTORY_BUFFER = torch.zeros((num_envs, history_len, 6), device=ee_pose.device, dtype=ee_pose.dtype)
        EE_HISTORY_LEN = history_len

    # 버퍼 shift (과거→앞으로 이동)
    EE_HISTORY_BUFFER = torch.roll(EE_HISTORY_BUFFER, shifts=-1, dims=1)
    EE_HISTORY_BUFFER[:, -1, :] = ee_pose  # 최신 값 업데이트

    # flatten해서 반환
    return EE_HISTORY_BUFFER.reshape(env.num_envs, history_len * 6)


def quat_to_euler(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quaternion → Euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat.unbind(dim=-1)

    # roll (x축 회전)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    # pitch (y축 회전)
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    # yaw (z축 회전)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return roll, pitch, yaw
