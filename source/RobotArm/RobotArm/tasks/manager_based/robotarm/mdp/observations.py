# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


from RobotArm.tasks.manager_based.robotarm.mdp.rewards import get_workpiece_size, get_ee_pose


# 전역 버퍼 (env 별 히스토리 저장용)
EE_HISTORY_BUFFER = None
EE_HISTORY_LEN = None


def grid_mask_state_obs(env, grid_mask_history_len=4):
    """
    Workpiece의 Grid Mask 상태 관찰
    """
    if not hasattr(env, "grid_x_num"):
        workpiece = env.scene["workpiece"]
        try:
            wp_size_x, wp_size_y = get_workpiece_size(workpiece)
        except Exception as e:
            wp_size_x, wp_size_y = 0.5, 0.5
            print(f"Workpiece size dynamic read failed in obs: {e}. Using default.")

        GRID_SIZE = 0.02
        grid_x_num = int(wp_size_x / GRID_SIZE)
        grid_y_num = int(wp_size_y / GRID_SIZE)
    else:
        grid_x_num = env.grid_x_num
        grid_y_num = env.grid_y_num

    # env.grid_mask가 정의되지 않았다면 초기화
    if not hasattr(env, "grid_mask"):
        total_dim = grid_x_num * grid_y_num * grid_mask_history_len

        return torch.zeros((env.num_envs, total_dim), device=env.device)
    
    # Grid Mask가 정의된 이후의 정상적인 실행 로직
    current_mask_float = env.grid_mask.float()
    is_reset = (env.episode_length_buf == 0).any()
    
    # 커버리지 비율 계산 및 디버깅 출력
    total_grids = grid_x_num * grid_y_num
    covered_counts = torch.sum(current_mask_float, dim=[1, 2])
    coverage_percentages = (covered_counts / total_grids) * 100
    first_env_coverage = coverage_percentages[0].item()
    print(f"Current Workpiece Coverage: {first_env_coverage:.2f}% (Total grids: {total_grids}, Covered: {covered_counts[0].item():.0f})")

    # 히스토리 버퍼 초기화 및 업데이트
    if not hasattr(env, "_grid_mask_history") or is_reset:
        history_shape = (env.num_envs, grid_mask_history_len) + current_mask_float.shape[1:]
        if is_reset and hasattr(env, "_grid_mask_history"):
            # 에피소드 리셋 시 0으로 초기화
            env._grid_mask_history = torch.zeros(history_shape, dtype=torch.float, device=env.device)
        else:
            # 환경 시작 시 초기화
            env._grid_mask_history = torch.stack([current_mask_float] * grid_mask_history_len, dim=1)
        
    # 새로운 Grid Mask 상태를 히스토리 큐에 추가 (FIFO)
    new_mask_float_unsqueeze = current_mask_float.unsqueeze(1)
    
    # 히스토리 업데이트: 가장 오래된 데이터 제거, 최신 데이터 추가
    env._grid_mask_history = torch.cat(
        [env._grid_mask_history[:, 1:], new_mask_float_unsqueeze], dim=1
    )

    # 관찰 벡터 형태로 평탄화 (Flatten)
    # shape: (num_envs, history_len * grid_x_num * grid_y_num)
    obs_vector = env._grid_mask_history.flatten(start_dim=1)
    
    return obs_vector



def ee_pose_history(env, asset_cfg: SceneEntityCfg, history_len: int = 5) -> torch.Tensor:
    """
    엔드이펙터 위치 및 자세(roll, pitch, yaw) 히스토리 반환
    """
    global EE_HISTORY_BUFFER, EE_HISTORY_LEN

    # ee_pose: (num_envs, 6) [x, y, z, roll, pitch, yaw]
    ee_pose = get_ee_pose(env, asset_name="robot") # (num_envs, 6)

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


# def get_contact_forces(env, sensor_name="contact_forces"):
#     """Mean contact wrench [Fx, Fy, Fz, 0, 0, 0]"""
#     sensor = env.scene.sensors[sensor_name]
#     forces_w = sensor.data.net_forces_w
#     mean_force = torch.mean(forces_w, dim=1)
#     zeros_torque = torch.zeros_like(mean_force)
#     contact_wrench = torch.cat([mean_force, zeros_torque], dim=-1)

#     step = int(env.common_step_counter)
#     if step % 100 == 0:
#         fx, fy, fz = mean_force[0].tolist()
#         print(f"[ContactSensor DEBUG] Step {step}: Fx={fx:.3f}, Fy={fy:.3f}, Fz={fz:.3f}")

#     return contact_wrench
