# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import RobotArm.tasks.manager_based.robotarm.mdp.path_manager as pm # 경로 매니저 import

# [1] 전역 버퍼 (속도/가속도 추정을 위한 히스토리 저장용)
EE_HISTORY_BUFFER = None
EE_HISTORY_LEN = None

# [2] 임시 경로 데이터 (Path Manager 대체)
# 로봇이 따라갈 점들의 좌표입니다. (X, Y, Z)
# 일단 직선 경로로 테스트하고, 나중에 txt 파일 로딩으로 바꿀 수 있습니다.
PATH_DATA = torch.tensor([
    [0.4, -0.2, 0.05],
    [0.4, -0.1, 0.05],
    [0.4,  0.0, 0.05],
    [0.4,  0.1, 0.05],
    [0.4,  0.2, 0.05],
    [0.5,  0.2, 0.05],
    [0.5,  0.1, 0.05],
    [0.5,  0.0, 0.05],
    [0.5, -0.1, 0.05],
    [0.5, -0.2, 0.05]
], dtype=torch.float32)

def path_tracking_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """
    [핵심] 로봇 손끝(EE) 위치와 목표 경로 사이의 오차 및 힘 센서 값을 관측합니다.
    """
    global PATH_DATA
    
    # 디바이스 설정 (CPU/GPU)
    device = env.device
    num_envs = env.num_envs
    
    # PATH_DATA를 현재 디바이스로 이동 (최초 1회만 수행됨)
    if PATH_DATA.device != device:
        PATH_DATA = PATH_DATA.to(device)
        
    # ------------------------------------------------------------------
    # 1. 현재 로봇 상태(EE 위치) 가져오기 (Import 없이 직접 계산)
    # ------------------------------------------------------------------
    # env.scene["robot"] 객체에서 바로 가져옵니다.
    # body_pos_w의 shape: (num_envs, num_bodies, 3)
    # UR10e의 경우 마지막 링크(-1)가 손끝(tool0)이라고 가정합니다.
    robot = env.scene["robot"]
    ee_pos = robot.data.body_pos_w[:, -1, :]      # 위치 (x, y, z)
    ee_quat = robot.data.body_quat_w[:, -1, :]    # 회전 (w, x, y, z)

    # ------------------------------------------------------------------
    # 2. 목표(Target) 찾기 - 가장 가까운 경로점 찾기
    # ------------------------------------------------------------------
    # (num_envs, 1, 3) - (1, num_points, 3) = (num_envs, num_points, 3)
    dists = torch.norm(ee_pos.unsqueeze(1) - PATH_DATA.unsqueeze(0), dim=2)
    
    # 각 환경별로 가장 가까운 점의 인덱스를 찾음
    min_dists, min_indices = torch.min(dists, dim=1)
    
    # 목표 위치 추출
    target_pos = PATH_DATA[min_indices] # (num_envs, 3)

    # ------------------------------------------------------------------
    # 3. 오차(Error) 계산
    # ------------------------------------------------------------------
    pos_error = target_pos - ee_pos

    # ------------------------------------------------------------------
    # 4. 힘(Force) 센서 데이터 가져오기
    # ------------------------------------------------------------------
    # EnvCfg에서 정의한 "contact_forces" 센서에 접근
    if "contact_forces" in env.scene.sensors:
        # net_forces_w: (num_envs, num_bodies, 3) -> 합력 계산
        # 보통 센서는 로봇 전체 링크에 달려있으므로, 그 중 '손'이나 '전체 합'을 구해야 함.
        # 여기서는 가장 강하게 작용하는 Z축 힘을 가져온다고 가정
        sensor_data = env.scene["contact_forces"].data.net_forces_w
        # 센서 데이터 중 Z축 힘(수직항력)의 절댓값의 최대값 or 합계 사용
        # (num_envs, num_links, 3) -> (num_envs, 1)
        force_z = torch.norm(sensor_data[..., 2], dim=-1, keepdim=True)
        
        # 값 클램핑 (너무 큰 값 방지)
        force_z = torch.clamp(force_z, 0.0, 50.0)
    else:
        force_z = torch.zeros((num_envs, 1), device=device)

    # ------------------------------------------------------------------
    # 5. 관측 벡터 합치기
    # ------------------------------------------------------------------
    # [위치오차(3), 힘(1), 현재위치(3)] = 총 7차원 (필요시 자세 오차 추가 가능)
    obs = torch.cat([pos_error, force_z, ee_pos], dim=-1)
    
    return obs


def ee_pose_history(env: ManagerBasedRLEnv, history_len: int = 5) -> torch.Tensor:
    """
    엔드이펙터의 위치 히스토리를 반환합니다. (속도감 학습용)
    """
    global EE_HISTORY_BUFFER, EE_HISTORY_LEN
    
    robot = env.scene["robot"]
    # 위치(3) + 회전(4) = 7
    current_pose = torch.cat([
        robot.data.body_pos_w[:, -1, :], 
        robot.data.body_quat_w[:, -1, :]
    ], dim=-1)

    # 버퍼 초기화 (처음 실행 시)
    if EE_HISTORY_BUFFER is None or EE_HISTORY_BUFFER.shape[0] != env.num_envs:
        num_envs = env.num_envs
        # (num_envs, history_len, 7)
        EE_HISTORY_BUFFER = torch.zeros(
            (num_envs, history_len, 7), 
            device=env.device, 
            dtype=current_pose.dtype
        )
        EE_HISTORY_LEN = history_len

    # 버퍼 시프트 (오래된 데이터 버리고 새 데이터 추가)
    # [0, 1, 2, 3, 4] -> [1, 2, 3, 4, 0] (마지막에 새거 넣음)
    EE_HISTORY_BUFFER = torch.roll(EE_HISTORY_BUFFER, shifts=-1, dims=1)
    EE_HISTORY_BUFFER[:, -1, :] = current_pose

    # Flatten해서 반환 (MLP에 넣기 위함)
    return EE_HISTORY_BUFFER.reshape(env.num_envs, -1)
