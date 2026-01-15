# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from RobotArm.tasks.manager_based.robotarm.mdp.rewards import get_ee_pose # rewards에서 EE 위치 가져오는 함수 재사용
import RobotArm.tasks.manager_based.robotarm.mdp.path_manager as pm # 경로 매니저 import

# 전역 버퍼 (속도/가속도 추정을 위한 히스토리 저장용)
EE_HISTORY_BUFFER = None
EE_HISTORY_LEN = None
# observations_1.py 수정 부분


def path_tracking_obs(env):
    """
    [핵심] 경로 추종 및 폴리싱을 위한 관측 함수
    
    Returns:
        obs (Tensor): [위치오차(3), 자세오차(3), 현재힘(1), 현재속도(6)] 등을 포함한 벡터
    """
    device = env.device
    num_envs = env.num_envs
    
    # ------------------------------------------------------------------
    # 1. 현재 로봇 상태 가져오기
    # ------------------------------------------------------------------
    # get_ee_pose는 [x, y, z, roll, pitch, yaw] 형태라고 가정
    ee_pose = get_ee_pose(env, asset_name="robot") 
    current_pos = ee_pose[:, :3]
    current_rot = ee_pose[:, 3:] # Euler angles (Roll, Pitch, Yaw)

    # ------------------------------------------------------------------
    # 2. 목표(Target) 가져오기 
    # *아직 CSV 로더가 없으므로, 임시로 'ㄹ'자 경로를 수식으로 생성합니다*
    # 나중에 이 부분을 env.targets[env.progress_idx] 로 교체하면 됩니다.
    # ------------------------------------------------------------------
    dt = env.step_dt
    t = env.episode_length_buf.float() * dt # 현재 시간
    
    # 임시 'ㄹ'자 경로 생성 (가로 0.5m, 세로 0.5m 영역)
    # X축: 좌우로 왕복 (Sine wave)
    # Y축: 앞으로 천천히 전진
    # Z축: 바닥 표면(0.05m) 유지
    #freq = 1.5 
    #target_x = 0.5 + 0.2 * torch.sin(freq * t) 
    #target_y = -0.3 + 0.05 * t
    #target_z = torch.full((num_envs,), 0.05, device=device) # 높이는 5cm로 고정
    
    #target_pos = torch.stack([target_x, target_y, target_z], dim=-1)
    
    # CSV 파일 기반 목표 가져오기
    target_pos = pm.get_target_pose_from_path(env)
    
    # 목표 자세: 항상 바닥을 수직으로 바라보기 (Roll=180도 or 0도, Pitch=0)
    # UR 로봇 기준, Tool이 아래를 보려면 보통 Rx=pi, Ry=0, Rz=0 등이 됨.
    # 여기서는 오차 계산을 위해 0으로 가정 (또는 env.default_rot 사용)
    target_rot = torch.zeros_like(current_rot) 


    # ------------------------------------------------------------------
    # 3. 오차(Error) 계산 (AI가 줄여야 할 값들)
    # ------------------------------------------------------------------
    pos_error = target_pos - current_pos
    rot_error = target_rot - current_rot # 단순 차이 (쿼터니언 사용 권장하지만 일단 오일러로)

    # ------------------------------------------------------------------
    # 4. 힘(Force) 센서 데이터 (중요!)
    # ------------------------------------------------------------------
    # Isaac Lab은 보통 env.contact_forces에 센서값이 들어옴
    if hasattr(env, "contact_forces"):
        # Z축 힘 (누르는 힘)만 추출. 
        # (센서 좌표계에 따라 Z가 아닐 수도 있으니 확인 필요, 보통 World Z)
        force_z = env.contact_forces[:, 2].unsqueeze(1)
        # 너무 튀는 값 방지 (Clamping)
        force_z = torch.clamp(force_z, -50.0, 50.0)
    else:
        # 센서가 없으면 0으로 채움
        force_z = torch.zeros((num_envs, 1), device=device)

    # ------------------------------------------------------------------
    # 5. 관측 벡터 합치기
    # ------------------------------------------------------------------
    # AI에게 주는 정보: [위치오차(3), 자세오차(3), 현재힘(1), 현재위치(3)]
    # 현재 위치를 주는 이유는 '내가 작업 공간 어디쯤 있는지' 알게 하기 위함
    obs = torch.cat([pos_error, rot_error, force_z, current_pos], dim=-1)
    
    return obs


def ee_pose_history(env, history_len: int = 5) -> torch.Tensor:
    """
    (기존 코드 유지) 움직임의 추세(Velocity/Acceleration)를 파악하기 위해 과거 기록을 봅니다.
    """
    global EE_HISTORY_BUFFER, EE_HISTORY_LEN

    ee_pose = get_ee_pose(env, asset_name="robot") 

    if EE_HISTORY_BUFFER is None or EE_HISTORY_LEN != history_len:
        num_envs = ee_pose.shape[0]
        EE_HISTORY_BUFFER = torch.zeros((num_envs, history_len, 6), device=ee_pose.device, dtype=ee_pose.dtype)
        EE_HISTORY_LEN = history_len

    EE_HISTORY_BUFFER = torch.roll(EE_HISTORY_BUFFER, shifts=-1, dims=1)
    EE_HISTORY_BUFFER[:, -1, :] = ee_pose 

    return EE_HISTORY_BUFFER.reshape(env.num_envs, history_len * 6)
