from __future__ import annotations

import torch
import math
import os
import csv
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_euler, quat_apply, quat_mul, quat_from_euler_xyz, euler_xyz_from_quat, combine_frame_transforms, quat_error_magnitude
from isaaclab.envs import ManagerBasedRLEnv
import carb.input # 키보드 입력용
from pxr import UsdGeom

# [중요] 데이터 로더 임포트
from . import path_loader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
if "nrs_fk_core" not in sys.modules:
    from nrs_fk_core import FKSolver
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver

from RobotArm.robots.ur10e_w_spindle import *

# --- [Target 생성 헬퍼 함수] ---
# observation_1.py와 완전히 동일한 로직을 써야, 로봇이 보는 목표와 채점 기준이 일치합니다.
def _get_generated_target(env):
    device = env.device
    num_envs = env.num_envs
    dt = env.step_dt
    t = env.episode_length_buf.float() * dt
    
    # 'ㄹ'자 경로 (obs와 동일 파라미터 유지 필수!)
    #freq = 1.5 
    #target_x = 0.5 + 0.2 * torch.sin(freq * t) 
    #target_y = -0.3 + 0.05 * t
    #target_z = torch.full((num_envs,), 0.05, device=device)
    
    #return torch.stack([target_x, target_y, target_z], dim=-1)
    
    #path_manager를 통해 CSV 경로 반환
    return pm.get_target_pose_from_path(env)


# --- [Helper Functions] ---

def get_ee_pose(env: ManagerBasedRLEnv, asset_name: str = "robot"):
    """
    Sim에서 제공하는 정확한 물리적 위치를 반환합니다.
    """
    robot = env.scene[asset_name]
    pos = robot.data.body_pos_w[:, -1, :]
    quat = robot.data.body_quat_w[:, -1, :]
    return torch.cat([pos, quat], dim=-1)

# --- [Reward Functions] ---

def track_path_reward(env: ManagerBasedRLEnv, sigma: float = 0.1):
    """
    [경로 추종] 현재 위치가 경로상의 가장 가까운 점과 얼마나 가까운지 평가
    """
    # 전역 변수 PATH_DATA 사용 (observations에서 가져옴)
    # 디바이스 동기화
    path_tensor = path_loader.get_path_tensor(env.device)
    if path_tensor.device != env.device:
        path_tensor = path_tensor.to(env.device)

    # 1. 현재 로봇 손끝 위치
    ee_pose = get_ee_pose(env)
    current_pos = ee_pose[:, :3] # (num_envs, 3)

    # 2. 경로상의 점들과의 거리 계산 (Broadcasting)
    # (num_envs, 1, 3) - (1, num_points, 3) -> (num_envs, num_points, 3)
    dists = torch.norm(current_pos.unsqueeze(1) - path_tensor.unsqueeze(0), dim=2)
    
    # 3. 가장 가까운 거리(Minimum Distance) 추출
    min_dist, _ = torch.min(dists, dim=1)

    # 4. 가우시안 커널 보상 (거리가 0이면 1.0, 멀어질수록 0.0)
    # log를 씌우지 않은 exp 형태라 0~1 사이 값 반환
    return torch.exp(-torch.square(min_dist) / (sigma ** 2))

def force_control_reward(env: ManagerBasedRLEnv, target_force: float = 10.0):
    """
    [힘 제어] Z축 접촉 힘을 목표값(10N)에 유지하면 보상
    """
    # 지금은 무조건 0.0을 반환해서 학습에 영향을 주지 않게 함
    return torch.zeros(env.num_envs, device=env.device)
    # --- 아래는 나중에 접촉 작업 할 때 주석 해제하세요 ---
    # 1. 센서 데이터 가져오기 (EnvCfg에 정의된 이름 확인 필수)
    #if "contact_forces" in env.scene.sensors:
     #   sensor = env.scene["contact_forces"]
        # net_forces_w: (num_envs, num_links, 3)
        # 2번 인덱스(Z축) 힘의 크기 사용
        # 보통 센서가 여러 링크에 걸쳐있을 수 있으니 합산하거나 특정 링크만 봐야 함
        # 여기서는 전체 링크 중 가장 큰 힘을 받는 곳 기준 or 합산
        # 안전하게: 마지막 차원(xyz) 중 z성분의 norm
      #  force_z = torch.abs(sensor.data.net_forces_w[..., 2])
        # 여러 링크 중 최대 힘 (보통 툴 끝)
      #  current_force, _ = torch.max(force_z, dim=-1) 
    #else:
    #    current_force = torch.zeros(env.num_envs, device=env.device)

    # 2. 오차 계산
#    force_error = torch.abs(current_force - target_force)

    # 3. 보상 변환 (오차가 0일 때 1.0)
    # 분모 1.0 더해서 0 나누기 방지
 #   return 1.0 / (1.0 + 0.1 * force_error)



def orientation_align_reward(env: ManagerBasedRLEnv):
    """
    [자세 유지] Tool이 바닥(World -Z 방향)을 수직으로 바라보는지 평가
    """
    ee_pose = get_ee_pose(env)
    ee_quat = ee_pose[:, 3:] # (qx, qy, qz, qw)

    # 1. 로봇 Tool의 Z축 벡터 (Local 0,0,1)
    tool_z_local = torch.zeros_like(ee_pose[:, :3])
    tool_z_local[:, 2] = 1.0 
    
    # 2. 쿼터니언 회전 적용 -> World 좌표계 벡터
    tool_z_world = quat_apply(ee_quat, tool_z_local)

    # 3. 목표 벡터 (World -Z: 0, 0, -1)
    target_dir = torch.zeros_like(tool_z_world)
    target_dir[:, 2] = -1.0 

    # 4. 내적 (Dot Product): 방향이 같으면 1.0
    dot_prod = torch.sum(tool_z_world * target_dir, dim=-1)
    
    return torch.clamp(dot_prod, min=0.0)
    
def action_smoothness_penalty(env: ManagerBasedRLEnv):
    """
    [부드러움] 급격한 동작 방지 (Action 값의 크기 억제)
    """
    # Action 값의 제곱합 -> 에너지를 적게 쓸수록(0에 가까울수록) 페널티 적음
    # 음수를 리턴하므로 페널티
    return -torch.sum(torch.square(env.action_manager.action), dim=-1)

def out_of_bounds_penalty(env: ManagerBasedRLEnv):
    """
    [이탈 방지] 작업 영역 벗어나면 벌점
    """
    # 1. 현재 위치
    ee_pos = get_ee_pose(env)[:, :3]
    
    # 2. 작업 영역 정의 (cfg에서 읽거나 기본값)
    # env.cfg가 아니라 env.unwrapped.cfg 등에 있을 수 있으므로 안전하게 getattr 사용
    # 여기서는 하드코딩된 안전 영역 설정
    wp_pos_x, wp_pos_y = 0.5, 0.0
    wp_size_x, wp_size_y = 0.6, 0.6 # 조금 여유 있게

    # 3. 경계 확인
    is_out_x = (ee_pos[:, 0] < (wp_pos_x - wp_size_x)) | (ee_pos[:, 0] > (wp_pos_x + wp_size_x))
    is_out_y = (ee_pos[:, 1] < (wp_pos_y - wp_size_y)) | (ee_pos[:, 1] > (wp_pos_y + wp_size_y))
    is_out_z = (ee_pos[:, 2] < 0.0) | (ee_pos[:, 2] > 0.5) # 바닥 뚫거나 너무 높이 가면

    is_out = (is_out_x | is_out_y | is_out_z).float()
    
    return -1.0 * is_out

def check_coverage_success(env: ManagerBasedRLEnv):
    """
    (종료 조건) 임시: 경로 오차가 아주 작으면 성공으로 간주?
    지금은 항상 False를 반환해서 시간 끝날 때까지 돌게 함
    """
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

def manual_termination(env: ManagerBasedRLEnv):
    """
    'K' 키를 누르면 강제 종료 (리셋)
    """
    try:
        input_i = carb.input.acquire_input_interface()
        keyboard = input_i.get_keyboard()
        if input_i.get_keyboard_value(keyboard, carb.input.KeyboardInput.K) > 0.5:
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    except Exception:
        pass # carb를 못 불러오거나 헤드리스 모드일 경우 무시
        
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


