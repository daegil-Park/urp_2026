from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_euler, combine_frame_transforms, quat_error_magnitude, quat_mul
from pxr import UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
if "nrs_fk_core" not in sys.modules:
    from nrs_fk_core import FKSolver
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver


from RobotArm.robots.ur10e_w_spindle import *


def get_workpiece_vertices(workpiece):
    """USD Mesh에서 Workpiece vertices 추출"""
    try:
        workpiece_prim = workpiece.prims[0]
        # Mesh Prim path 찾기: usd 파일에 따라 변경 필요
        mesh_prim = workpiece_prim.GetChild("World").GetChild("flat_surface_5").GetChild("mesh_").GetChild("Mesh")
        if not mesh_prim:
            raise ValueError("Mesh Prim not found at the final path.")

        mesh = UsdGeom.Mesh(mesh_prim)
        vertices = mesh.GetPointsAttr().Get()
        if vertices is None:
            raise ValueError("Vertices data is None. USD Mesh points not loaded or found.")
        
        return vertices
        
    except Exception as e:
        print(f"[get_workpiece_size] USD size read failed: {e}, using default size 0.5x0.5")
        return None
    

def get_workpiece_size(workpiece):
    """USD Mesh에서 Workpiece 크기 추출, 실패 시 기본값 0.5x0.5 사용"""
    wp_size_x, wp_size_y = 0.5, 0.5
    vertices = get_workpiece_vertices(workpiece)
    if vertices is None:
        return wp_size_x, wp_size_y
    else:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        wp_size_x = max(xs) - min(xs)
        wp_size_y = max(ys) - min(ys)

        return wp_size_x, wp_size_y


def get_workpiece_surface_height(workpiece, surface_offset=0.005):
    """
    Workpiece 표면의 월드 Z 좌표 계산
    """
    vertices = get_workpiece_vertices(workpiece)
    if vertices is None:
        print(f"[get_surface_height] Failed to determine Z height: {e}, using fallback height 0.0955")
        return 0.0955
    else:
        zs = [v[2] for v in vertices]
        workpiece_z_size = max(zs) - min(zs)
        target_height = workpiece_z_size + surface_offset
        return target_height


def get_ee_pose(env: "ManagerBasedRLEnv", asset_name: str = "robot"):
    """
    Returns end-effector pose (x, y, z, roll, pitch, yaw)
    -----------------------------------------------------
    - 현재 로봇의 joint 상태(q1~q6)를 불러와서
      FKSolver를 이용해 FK 계산 수행
    - FK 결과는 torch.Tensor (num_envs, 6) 형태로 반환
    """
    robot = env.scene[asset_name]
    q = robot.data.joint_pos[:, :6]  # (num_envs, 6)
    num_envs = q.shape[0]

    fk_solver = FKSolver(tool_z=0.239, use_degrees=False)
    ee_pose_list = []

    for i in range(num_envs):
        q_np = q[i].cpu().numpy().astype(float)
        ok, pose = fk_solver.compute(q_np, as_degrees=False)
        if not ok:
            ee_pose_list.append([float('nan')]*6)
        else:
            ee_pose_list.append([pose.x, pose.y, pose.z, pose.r, pose.p, pose.yaw])

    ee_pose = torch.tensor(ee_pose_list, dtype=torch.float32, device=q.device)
    assert ee_pose.ndim == 2 and ee_pose.shape[1] == 6, f"[EE_POSE] Invalid shape: {ee_pose.shape}"

    return ee_pose


def ee_to_grid(env, ee_frame_name=EE_FRAME_NAME, grid_size=0.02):
    """
    EE 좌표를 grid 좌표로 변환
    """
    ee_pose = get_ee_pose(env, asset_name="robot") 
    ee_pos = ee_pose[:, :3] # 위치 (x, y, z)만 사용

    workpiece = env.scene["workpiece"]
    workpiece_pos_tensor, _ = workpiece.get_world_poses()
    workpiece_pos_tensor = workpiece_pos_tensor.to(env.device)
    wp_pos = workpiece_pos_tensor.squeeze()[:3]
    
    if not hasattr(env, "wp_size_x"):
        wp_size_x, wp_size_y = get_workpiece_size(workpiece)
        env.wp_size_x = wp_size_x
        env.wp_size_y = wp_size_y
        env.grid_x_num = int(wp_size_x / grid_size)
        env.grid_y_num = int(wp_size_y / grid_size)
    
    wp_size_x_t = torch.tensor(env.wp_size_x, device=env.device)
    wp_size_y_t = torch.tensor(env.wp_size_y, device=env.device)
    grid_x_num = env.grid_x_num
    grid_y_num = env.grid_y_num

    origin_x = wp_pos[0] - wp_size_x_t / 2
    origin_y = wp_pos[1] - wp_size_y_t / 2

    ee_xy = ee_pos[:, :2].clone()
    ee_xy[:, 0] -= origin_x
    ee_xy[:, 1] -= origin_y

    grid_x = torch.clamp((ee_xy[:, 0] / grid_size).long(), 0, grid_x_num - 1)
    grid_y = torch.clamp((ee_xy[:, 1] / grid_size).long(), 0, grid_y_num - 1)

    return grid_x, grid_y


def coverage_reward(env, grid_size=0.02):
    """
    엔드이펙터가 방문한 grid cell의 증가분을 계산해 coverage reward를 반환.
    """
    grid_x, grid_y = ee_to_grid(env, grid_size=grid_size)
    total_cells = env.grid_x_num * env.grid_y_num
    
    # grid mask 초기화
    if not hasattr(env, "grid_mask"):
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    previous_mask = env.grid_mask.clone()
    
    # 현재 위치 방문 표시: 새로 방문한 셀 = 1, 재방문/미방문 셀 = 0
    indices = torch.arange(env.num_envs, device=env.device)
    env.grid_mask[indices, grid_x, grid_y] = True

    # 새로 방문한 셀 수
    newly_visited = (env.grid_mask.long() - previous_mask.long()).sum(dim=(1, 2)).float()
    
    # 커버리지 비율 (0.0 ~ 1.0)
    current_covered_count = env.grid_mask.sum(dim=(1, 2)).float()
    coverage_ratio = current_covered_count / total_cells

    # exponential scaling 적용
    exp_reward = newly_visited * (torch.exp(3.0 * coverage_ratio) - 1.0)

    return exp_reward


def revisit_penalty(env, grid_size=0.02):
    """
    이미 방문한 grid cell을 다시 방문하면 벌점
    """
    grid_x, grid_y = ee_to_grid(env, grid_size=grid_size)

    # grid_mask 초기화
    if not hasattr(env, "grid_mask"):
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    # 현재 EE 위치가 grid_mask에서 True인지 확인 (재방문 여부 확인)
    indices = torch.arange(env.num_envs, device=env.device)
    is_revisited = env.grid_mask[indices, grid_x, grid_y]
    revisited = is_revisited.float()   # True -> 1.0, False -> 0.0

    return -revisited


def coverage_completion_reward(env, threshold=0.95, bonus_scale=10.0):
    """
    surface coverage 비율이 threshold를 넘으면 bonus_scale 만큼의 보상 제공
    """
    if not hasattr(env, "grid_mask"):
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    num_envs = env.grid_mask.shape[0]
    # 현재 커버리지 비율 (0.0 ~ 1.0)
    completion = env.grid_mask.view(env.num_envs, -1).float().mean(dim=1)
    
    # 임계값(threshold)을 넘어선 부분만 추출
    over_threshold = torch.clamp(completion - threshold, min=0.0)
    # 남은 완료 비율 (1 - threshold)로 나누어 다시 0.0 ~ 1.0 범위로 정규화
    # (1.0 - threshold)가 0에 가까우면 나눗셈이 불안정해질 수 있으므로 epsilon 추가
    remaining_range = 1.0 - threshold
    epsilon = 1e-6
    if remaining_range > epsilon:
        normalized_bonus_ratio = over_threshold / remaining_range
    else:
        # threshold가 거의 1.0인 경우, over_threshold가 0보다 크면 1.0을 반환
        normalized_bonus_ratio = (over_threshold > 0.0).float()
    
    # 커버리지 100% 달성 시 bonus_scale 만큼의 보상
    return normalized_bonus_ratio * bonus_scale


def reset_grid_mask(env, env_ids):
    """
    에피소드 리셋 시 env.grid_mask 텐서를 False(0)로 초기화하고,
    최초 호출 시 Grid Mask를 생성
    """
    workpiece = env.scene["workpiece"]
    GRID_SIZE = 0.02

    # Grid 차원 정보가 env에 없으면 계산 및 저장
    if not hasattr(env, "wp_size_x"):
        wp_size_x, wp_size_y = get_workpiece_size(workpiece)
        env.wp_size_x = wp_size_x
        env.wp_size_y = wp_size_y
        env.grid_x_num = int(wp_size_x / GRID_SIZE)
        env.grid_y_num = int(wp_size_y / GRID_SIZE)

    # 2. grid_mask 속성이 env에 없으면 초기 생성
    if not hasattr(env, "grid_mask"):
        env.grid_mask = torch.zeros((env.num_envs, env.grid_x_num, env.grid_y_num), dtype=torch.bool, device=env.device)

    # reset시 grid_mask 초기화
    env.grid_mask[env_ids] = False
    
    # Grid Mask 상태 관찰(obs)에서 사용하는 히스토리도 함께 초기화
    if hasattr(env, "_grid_mask_history"):
        env._grid_mask_history[env_ids] = 0.0

    return {}


def surface_proximity_reward(env, asset_cfg: SceneEntityCfg):
    ee_pose = get_ee_pose(env, asset_name="robot") 
    ee_z = ee_pose[:, 2]

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
    ee_pose = get_ee_pose(env, asset_name="robot") 
    roll, pitch, yaw = ee_pose[:, 3], ee_pose[:, 4], ee_pose[:, 5]
    
    # 엔드이펙터 Z축 벡터 계산
    ee_z_axis_w = matrix_from_euler(roll, pitch, yaw) # shape: (num_envs, 3)
    
    # 목표 축
    target_axis_t = torch.tensor(target_axis, dtype=ee_z_axis_w.dtype, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    
    # 내적(Dot product) 계산: alignment_measure = cos(theta)
    alignment_measure = torch.abs(torch.sum(ee_z_axis_w * target_axis_t, dim=1))
    
    # 보상: 1에 가까울수록 높은 보상
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
