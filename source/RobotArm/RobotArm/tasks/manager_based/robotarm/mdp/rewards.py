from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, combine_frame_transforms, quat_error_magnitude, quat_mul
from pxr import UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from RobotArm.robots.ur10e_w_spindle import *

# def get_mesh_prim_path(workpiece):

def get_workpiece_size(workpiece):
    """USD Mesh에서 Workpiece 크기 추출, 실패 시 기본값 0.5x0.5 사용"""
    wp_size_x, wp_size_y = 0.5, 0.5
    try:
        workpiece_prim = workpiece.prims[0]
        
        # Mesh Prim path 찾기: usd 파일에 따라 변경 필요
        actual_mesh_prim = workpiece_prim.GetChild("World").GetChild("flat_surface_5").GetChild("mesh_").GetChild("Mesh")
        if not actual_mesh_prim:
            raise ValueError("Mesh Prim not found at the final path.")
            
        # Mesh Prim을 사용하여 데이터 읽기
        mesh = UsdGeom.Mesh(actual_mesh_prim)
        
        # UsdGeom.Mesh에서 정점 데이터 가져오기
        vertices = mesh.GetPointsAttr().Get()

        # NoneType 체크 및 크기 계산 (이전 로직과 동일)
        if vertices is None:
            raise ValueError("Vertices data is None. USD Mesh points not loaded or found.")
        
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        wp_size_x = max(xs) - min(xs)
        wp_size_y = max(ys) - min(ys)
        # print(f"Workpiece size: x={wp_size_x}, y={wp_size_y}")
        
    except Exception as e:
        print(f"[get_workpiece_size] USD size read failed: {e}, using default size 0.5x0.5")
        
    return wp_size_x, wp_size_y


def get_workpiece_surface_height(workpiece, surface_offset=0.005):
    """
    Workpiece의 Prim에서 메쉬 위치를 탐색하고, Workpiece 표면의 월드 Z 좌표를 계산합니다.
    (환경 초기화 시점에 단 한 번 호출되어야 합니다.)

    Args:
        workpiece_prim (Usd.Prim): Workpiece Asset의 최상위 Prim (예: /Workpiece).
        surface_offset (float): Workpiece 지오메트리 Z 위치에 더할 엔드이펙터의 이상적인 이격 거리 (m).

    Returns:
        float: Workpiece 표면의 월드 Z 좌표 + 이격 거리 (target_height).
    """
    try:
        workpiece_prim = workpiece.prims[0]
        
        # Mesh Prim path 찾기: usd 파일에 따라 변경 필요
        actual_mesh_prim = workpiece_prim.GetChild("World").GetChild("flat_surface_5").GetChild("mesh_").GetChild("Mesh")
        if not actual_mesh_prim:
            raise ValueError("Mesh Prim not found at the final path.")
            
        # Mesh Prim을 사용하여 데이터 읽기
        mesh = UsdGeom.Mesh(actual_mesh_prim)
        
        # UsdGeom.Mesh에서 정점 데이터 가져오기
        vertices = mesh.GetPointsAttr().Get()

        # NoneType 체크 및 크기 계산 (이전 로직과 동일)
        if vertices is None:
            raise ValueError("Vertices data is None. USD Mesh points not loaded or found.")

        zs = [v[2] for v in vertices]
        workpiece_z_size = max(zs) - min(zs)
        target_height = workpiece_z_size + surface_offset
        
        # print(f"Workpiece Center Z Position (World): {workpiece_z_size:.4f}m")
        # print(f"Calculated Target Surface Height: {target_height:.4f}m")
        
        return target_height
    
    except Exception as e:
        print(f"[get_surface_height] Failed to determine Z height: {e}, using fallback height 0.0955")
        # 실패 시 fallback 값 반환
        return 0.0955
    

def ee_to_grid(env, ee_frame_name=EE_FRAME_NAME, grid_size=0.02):
    """
    EE 좌표를 grid 좌표로 변환 (env.grid_x, env.grid_y 사용)
    """
    ee_index = env.scene["robot"].body_names.index(ee_frame_name)
    ee_pos = env.scene["robot"].data.body_pos_w[:, ee_index]

    workpiece = env.scene["workpiece"]
    workpiece_pos_tensor, _ = workpiece.get_world_poses()
    workpiece_pos_tensor = workpiece_pos_tensor.to(env.device)
    # Workpiece 월드 위치
    wp_pos = workpiece_pos_tensor.squeeze()[:3]
    
    
    wp_size_x, wp_size_y = get_workpiece_size(workpiece)
    wp_size_x_t = torch.tensor(wp_size_x, device=env.device)
    wp_size_y_t = torch.tensor(wp_size_y, device=env.device)

    origin_x = wp_pos[0] - wp_size_x_t / 2
    origin_y = wp_pos[1] - wp_size_y_t / 2

    ee_xy = ee_pos[:, :2].clone()
    ee_xy[:, 0] -= origin_x
    ee_xy[:, 1] -= origin_y

    grid_x_num = int(wp_size_x / grid_size)
    grid_y_num = int(wp_size_y / grid_size)
    grid_x = torch.clamp((ee_xy[:, 0] / grid_size).long(), 0, grid_x_num - 1)
    grid_y = torch.clamp((ee_xy[:, 1] / grid_size).long(), 0, grid_y_num - 1)

    num_envs = ee_pos.shape[0]
    return ee_pos, wp_size_x, wp_size_y, grid_x, grid_y, num_envs, grid_x_num, grid_y_num


def coverage_reward(env, grid_size=0.02):
    """
    엔드이펙터가 방문한 grid cell의 증가분을 계산해 coverage reward를 반환.
    """
    ee_pos, wp_size_x, wp_size_y, grid_x, grid_y, num_envs, grid_x_num, grid_y_num = ee_to_grid(env, grid_size=grid_size)
    total_cells = grid_x_num * grid_y_num
    
    # grid mask 초기화
    num_envs = ee_pos.shape[0]
    if not hasattr(env, "grid_mask"):
        env.grid_mask = torch.zeros((num_envs, grid_x_num, grid_y_num), dtype=torch.bool, device=env.device)
        newly_visited = torch.zeros(num_envs, dtype=torch.float, device=env.device)
    else:
        previous_mask = env.grid_mask.clone()
        
        # 현재 위치 방문 표시: 새로 방문한 셀 = 1, 재방문/미방문 셀 = 0
        indices = torch.arange(num_envs, device=env.device)
        env.grid_mask[indices, grid_x, grid_y] = True
        # 새로 방문한 셀 수
        newly_visited = (env.grid_mask.long() - previous_mask.long()).sum(dim=(1, 2)).float()
    
    # 커버리지 비율 (0.0 ~ 1.0) 계산
    current_covered_count = env.grid_mask.sum(dim=(1, 2)).float()
    coverage_ratio = current_covered_count / total_cells

    # exponential scaling 적용
    exp_reward = newly_visited * (torch.exp(3.0 * coverage_ratio) - 1.0)

    return exp_reward


def revisit_penalty(env, grid_size=0.02):
    """
    이미 방문한 grid cell을 다시 방문하면 벌점.
    last_ee_cell: 이전 step의 EE grid 위치 (num_envs, 2)
    """
    ee_pos, wp_size_x, wp_size_y, grid_x, grid_y, num_envs, grid_x_num, grid_y_num = ee_to_grid(env, grid_size=grid_size)

    # grid_mask 초기화
    if env.grid_mask is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 현재 EE 위치가 grid_mask에서 True인지 확인 (재방문 여부 확인)
    indices = torch.arange(num_envs, device=env.device)
    is_revisited = env.grid_mask[indices, grid_x, grid_y]
    revisited = is_revisited.float()   # True -> 1.0, False -> 0.0

    return -revisited


# def coverage_completion_reward(env, threshold=0.95, bonus_scale=10.0):
#     """
#     surface coverage 비율이 threshold를 넘으면 bonus_scale 만큼의 보상 제공
#     """
#     workpiece = env.scene["workpiece"]

#     GRID_SIZE = 0.02
#     wp_size_x, wp_size_y = get_workpiece_size(workpiece)
#     grid_x_num = int(wp_size_x / GRID_SIZE)
#     grid_y_num = int(wp_size_y / GRID_SIZE)

#     if not hasattr(env, "grid_mask"):
#         print("Grid Mask not initialized; returning dummy tensor. (First run)")
#         total_dim = grid_x_num * grid_y_num
#         return torch.zeros((env.num_envs, total_dim), device=env.device)
    
#     num_envs = env.grid_mask.shape[0]
#     # 현재 커버리지 비율 (0.0 ~ 1.0)
#     completion = env.grid_mask.view(num_envs, -1).float().mean(dim=1)
#     # 임계값(threshold)을 넘어선 부분만 추출
#     over_threshold = torch.clamp(completion - threshold, min=0.0)

#     # (1.0 - threshold)가 0에 가까우면 나눗셈이 불안정해질 수 있으므로 epsilon 추가
#     remaining_range = 1.0 - threshold
#     epsilon = 1e-6
#     if remaining_range > epsilon:
#         normalized_bonus_ratio = over_threshold / remaining_range
#     else:
#         # threshold가 거의 1.0인 경우, over_threshold가 0보다 크면 1.0을 반환
#         normalized_bonus_ratio = (over_threshold > 0.0).float()
    
#     # 커버리지 100% 달성 시 bonus_scale 만큼의 보상
#     return normalized_bonus_ratio * bonus_scale


def reset_grid_mask(env, env_ids):
    """
    에피소드 리셋 시 env.grid_mask 텐서를 False(0)로 초기화합니다.
    """
    # grid_mask의 존재를 확인하고 초기화
    if hasattr(env, "grid_mask"):
        # 마스크를 0(False)으로 설정
        env.grid_mask[env_ids] = False
    
    # Grid Mask 상태 관찰(obs)에서 사용하는 히스토리도 함께 초기화
    if hasattr(env, "_grid_mask_history"):
        env._grid_mask_history[env_ids] = 0.0

    return {}


def surface_proximity_reward(env, asset_cfg: SceneEntityCfg):
    # 엔드이펙터(EE)의 현재 위치
    asset = env.scene[asset_cfg.name]
    ee_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    
    # workpiece의 높이(평면 가정)
    workpiece = env.scene["workpiece"]
    target_z = get_workpiece_surface_height(workpiece, surface_offset=0.005)
    target_z_tensor = torch.full_like(ee_z, target_z)
    error = torch.abs(ee_z - target_z_tensor)
    
    return torch.exp(-10 * error)


def ee_orientation_alignment(env, asset_cfg: SceneEntityCfg, target_axis=(0.0, 0.0, -1.0)):
    """
    엔드이펙터(EE)의 Z축이 월드 좌표계의 목표 축(Target Axis)과 정렬되도록 보상을 제공
    (폴리싱 작업에서 Workpiece 표면에 수직을 유지하기 위함)
    """
    # 1. EE의 월드 회전(쿼터니언) 가져오기
    ee_asset = env.scene[asset_cfg.name]
    ee_quat_w = ee_asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    
    # 2. EE의 Z축(툴팁 방향) 월드 벡터 계산
    rot_matrix = matrix_from_quat(ee_quat_w) # (num_envs, 3, 3)
    # EE의 Z축 (로컬 (0, 0, 1))은 회전 행렬의 세 번째 열 벡터입니다.
    ee_z_axis_w = rot_matrix[:, :, 2] # shape: (num_envs, 3)
    
    # 3. 목표 축 텐서 생성
    target_axis_t = torch.tensor(target_axis, dtype=ee_z_axis_w.dtype, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    
    # 4. 정렬 보상 계산 (코사인 유사도 사용)
    # EE Z축 벡터와 목표 축 벡터 간의 내적 (Dot product)을 계산합니다.
    # |cos(theta)| = |A · B| / (|A| |B|). EE Z축과 목표 축은 단위 벡터이므로, 내적은 유사도입니다.
    # 수직(Parallel, theta=0)이면 1, 수평(Perpendicular, theta=90)이면 0.
    alignment_measure = torch.abs(torch.sum(ee_z_axis_w * target_axis_t, dim=1))
    
    # 5. 보상 반환: 1에 가까울수록 높은 보상
    return alignment_measure


def time_efficiency_reward(env, max_steps: int = 1000):
    """
    시간 효율성에 대한 보상함수
    """
    # 현재 step
    current_step_tensor = env.episode_length_buf.to(torch.float32)
    max_steps_tensor = torch.tensor(max_steps, dtype=torch.float32, device=env.device)

    # normalize: 0 ~ 1, 작을수록 reward 높음
    reward = 1.0 - (current_step_tensor / max_steps_tensor)
    reward = torch.clamp(reward, min=0.0, max=1.0)

    return reward
