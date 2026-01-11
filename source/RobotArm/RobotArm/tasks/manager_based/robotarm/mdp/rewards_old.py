from __future__ import annotations

import torch
import os
import csv
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_euler, quat_apply, quat_mul, quat_from_euler_xyz, euler_xyz_from_quat, combine_frame_transforms, quat_error_magnitude
from pxr import UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
if "nrs_fk_core" not in sys.modules:
    from nrs_fk_core import FKSolver
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver


from RobotArm.robots.ur10e_w_spindle import *

GRID_SIZE = 0.1 # 10cm 그리드 셀 크기
ENV_ID = 10     # 디버깅용 환경 인덱스

# def new_visit_reward(env: "ManagerBasedRLEnv"):
#     """
#     새로 방문한 grid cell의 증가분을 계산해 보상을 반환합니다.
#     (Coverage Exploration의 즉각적인 보상 요소)
#     """
#     # grid_mask 업데이트는 항상 필요합니다.
#     grid_x, grid_y = ee_to_grid(env)

#     if not hasattr(env, "grid_mask"):
#         return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

#     previous_mask = env.grid_mask.clone()
#     indices = torch.arange(env.num_envs, device=env.device)
    
#     # 현재 위치 방문 표시 (마스크 업데이트)
#     env.grid_mask[indices, grid_x, grid_y] = True

#     # 새로 방문한 셀 수 (env별)
#     newly_visited = (env.grid_mask.long() - previous_mask.long()).sum(dim=(1, 2)).float()

#     return newly_visited


def coverage_reward(env: "ManagerBasedRLEnv", exp_scale: float = 5.0):
    """
    엔드이펙터가 방문한 grid cell의 증가분을 계산해 coverage reward를 반환.
    """
    grid_x, grid_y = ee_to_grid(env)
    
    if not hasattr(env, "grid_mask"):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    indices = torch.arange(env.num_envs, device=env.device)
    total_cells = env.grid_x_num * env.grid_y_num

    # 업데이트 전의 coverage 저장
    previous_covered_count = env.grid_mask.sum(dim=(1, 2)).float()

    # 이 비율을 env에 저장 (다음 스텝의 revisit_penalty가 이 값을 사용합니다)
    env._prev_covered_count = previous_covered_count.clone()

    # 현재 위치 방문 표시 (마스크 업데이트)
    env.grid_mask[indices, grid_x, grid_y] = True
    
    # 전체 coverage 비율
    current_covered_count = env.grid_mask.sum(dim=(1, 2)).float()
    coverage_ratio = current_covered_count / float(total_cells)  # (num_envs,)

    env._current_covered_count = current_covered_count.clone()

    print(f"Mean Workpiece Coverage: {coverage_ratio.mean().item()*100:.2f}% "
          f"(Total cells: {total_cells}, Covered: {current_covered_count.mean().item():.0f})")
    # print(f"Current Workpiece Coverage: {coverage_ratio[ENV_ID].item()*100:.2f}% "
    #       f"(Total cells: {total_cells}, Covered: {current_covered_count[ENV_ID].item():.0f})")
    log_coverage_data(env, coverage_ratio[ENV_ID])

    # exponential scaling 적용
    exp_coverage_reward = torch.exp(exp_scale * coverage_ratio)

    return exp_coverage_reward


def ee_movement_reward(env: "ManagerBasedRLEnv", max_movement: float = 0.05):
    """
    엔드 이펙터의 움직인 거리에 대한 보상을 제공하여 정지 정책을 방지합니다.
    (Exploration 유도 및 Inaction Penalty 대체 요소)
    """
    ee_pose = get_ee_pose(env, asset_name="robot")
    ee_pos = ee_pose[:, :3]

    if not hasattr(env, "_prev_ee_pos"):
        # 초기화. 다음 스텝부터 유효합니다.
        env._prev_ee_pos = ee_pos.clone()
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # 움직인 거리 계산
    movement = torch.norm(ee_pos - env._prev_ee_pos, dim=1)  # (num_envs,)
    env._prev_ee_pos = ee_pos.clone()

    # movement를 클리핑하여 너무 격렬한 움직임에 과도한 보상을 주지 않도록 합니다.
    movement_clipped = torch.clamp(movement, 0.0, max_movement)

    return movement_clipped


def out_of_bounds_penalty(env: "ManagerBasedRLEnv"):
    """
    작업물 XY 범위를 벗어난 경우 강한 패널티를 부여하는 보상 함수.
    scale: penalty strength (기본=5.0)
    """
    robot = env.scene["robot"]
    workpiece = env.scene["workpiece"]

    base_pos_w = robot.data.root_pos_w  # 로봇 베이스의 월드 위치 (Num_Envs, 3)
    WORKPIECE_REL_POS = torch.tensor([0.75, 0.0, 0.0], dtype=base_pos_w.dtype, device=env.device)
    wp_pos = base_pos_w + WORKPIECE_REL_POS.unsqueeze(0)
    
    ee_pos = get_ee_pose(env, asset_name="robot")[:, :3]
    ee_x = ee_pos[:, 0]
    ee_y = ee_pos[:, 1]

    wp_size_x = env.wp_size_x
    wp_size_y = env.wp_size_y

    half_x = wp_size_x / 2
    half_y = wp_size_y / 2

    # 유효 범위
    min_x = wp_pos[:, 0] - half_x
    max_x = wp_pos[:, 0] + half_x
    min_y = wp_pos[:, 1] - half_y
    max_y = wp_pos[:, 1] + half_y

    # 범위 검사
    out_x = (ee_x < min_x) | (ee_x > max_x)
    out_y = (ee_y < min_y) | (ee_y > max_y)
    out_of_bounds = (out_x | out_y).float()

    # --- 누적 카운터 ---
    if not hasattr(env, "_out_of_bounds_count"):
        env._out_of_bounds_count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    env._out_of_bounds_count += out_of_bounds
    # 에피소드 리셋 시 카운터 초기화
    env._out_of_bounds_count[env.episode_length_buf == 0] = 0.0
    # 누적 횟수에 따른 패널티 계산 (ex: 선형, 혹은 지수적 증가)
    count_based_penalty = env._out_of_bounds_count * 0.1  # 매번 이탈 시 패널티 0.5씩 증가

    # 실제 패널티 (이탈이 발생한 경우에만 적용)
    final_penalty = -out_of_bounds * count_based_penalty

    return final_penalty


def revisit_penalty(env: "ManagerBasedRLEnv"):
    """
    이미 방문한 grid cell을 다시 방문하면 벌점
    """
    prev_covered_count = env._prev_covered_count
    current_covered_count = env._current_covered_count
    no_new_coverage = (current_covered_count == prev_covered_count)
                       
    if not hasattr(env, "_revisit_count"):
        env._revisit_count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    revisit_increment = no_new_coverage.float()
    env._revisit_count += revisit_increment

    count_based_penalty = env._revisit_count * 0.1  # 매번 중복 방문 시 패널티 0.1씩 증가
    final_penalty = no_new_coverage.float() * (-count_based_penalty)

    return final_penalty


def surface_proximity_reward(env: "ManagerBasedRLEnv"):
    ee_pose = get_ee_pose(env, asset_name="robot") 
    ee_z = ee_pose[:, 2]

    # workpiece의 높이(평면 가정)
    workpiece = env.scene["workpiece"]
    target_z = get_workpiece_surface_height(workpiece, surface_offset=0.239)
    target_z_tensor = torch.full_like(ee_z, target_z)
    error = torch.abs(ee_z - target_z_tensor)

    error = torch.clamp(error - 0.05, min=0)  # 5cm 이내 오차는 허용
    return torch.exp(-3 * error)


def ee_orientation_alignment(env: "ManagerBasedRLEnv", target_axis=(0.0, 0.0, -1.0)):
    """
    엔드이펙터(EE)의 Z축이 월드 좌표계의 목표 축(Target Axis)과 정렬되도록 보상을 제공
    """
    ee_pose = get_ee_pose(env, asset_name="robot") 
    roll, pitch, yaw = ee_pose[:, 3], ee_pose[:, 4], ee_pose[:, 5]
    
    # (num_envs, 3) 텐서로 결합, 회전 행렬 계산
    euler_angles = torch.stack([roll, pitch, yaw], dim=-1)
    rot_mat = matrix_from_euler(euler_angles, "XYZ")
    ee_z_axis_w = rot_mat[:, :, 2]
    
    # 목표 축과 비교
    target_axis_t = torch.tensor(target_axis, dtype=ee_z_axis_w.dtype, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    
    # 내적(Dot product)으로 정렬도 계산 (1에 가까울수록 수직)
    alignment_measure = torch.sum(ee_z_axis_w * target_axis_t, dim=1)
    alignment_reward = (alignment_measure + 1.0) / 2.0
    # alignment_reward = torch.clamp(alignment_measure, min=0.0)
    
    return alignment_reward


def time_efficiency_reward(env: "ManagerBasedRLEnv", max_steps: int = 1000):
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


def distance_to_workpiece_reward(env: "ManagerBasedRLEnv"):
    """
    가장자리에 있는 로봇을 작업물 중앙으로 당겨오는 자석 보상
    """
    # 1. 로봇 손끝(EE) 위치 (X, Y)
    ee_pose = get_ee_pose(env, asset_name="robot")
    ee_xy = ee_pose[:, :2]

    # 2. 작업물 중심 위치 (X, Y)
    workpiece = env.scene["workpiece"]
    workpiece_pos_tensor, _ = workpiece.get_world_poses()
    workpiece_pos_tensor = workpiece_pos_tensor.to(env.device)
    target_xy = workpiece_pos_tensor.squeeze()[:2]

    # 3. 거리 계산 (멀수록 점수가 작아짐)
    distance = torch.norm(ee_xy - target_xy, dim=1)
    
    # 4. 거리가 0에 가까우면 1점, 멀면 0점
    return torch.exp(-2.0 * distance)


# ====================================================
# 종료 조건 함수
# ====================================================

def check_coverage_success(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Workpiece 커버리지가 100%인지 확인하고 성공 텐서를 반환합니다.
    """
    if not hasattr(env, "grid_mask"):
        # grid_mask가 없으면 아직 성공할 수 없음
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    total_cells = env.grid_x_num * env.grid_y_num
    
    # 현재 방문한 셀의 수 (env별)
    current_covered_count = env.grid_mask.sum(dim=(1, 2)).float()
    
    # current_covered_count가 total_cells과 같거나 큰 경우 성공 (85% 달성)
    is_success = (current_covered_count/total_cells >= 0.95)
    
    return is_success


# ====================================================
# 유틸리티 함수들
# ====================================================

LOG_FILE = "coverage_history.csv"
LOG_HEADER = ["step", "coverage_ratio"]

def init_coverage_logger():
    """CSV 로깅 파일을 초기화하고 헤더 작성"""
    # 기존 파일이 있다면 삭제
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(LOG_HEADER)

def log_coverage_data(env, coverage_ratio):
    """
    특정 환경(env_id)의 커버율 데이터를 CSV 파일에 추가
    """
    # 텐서의 데이터를 CPU로 옮겨 Python float으로 변환
    current_ratio = coverage_ratio.item()
    # 현재 에피소드 및 스텝 정보 가져오기 (env.episode_length_buf 사용)
    current_step = env.episode_length_buf[ENV_ID].item()

    row = [current_step, current_ratio * 100]
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

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
        print(f"USD read failed: {e}, returning None for vertices.")
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
        wp_size_x = max(xs) - min(xs)# EE의 월드 XY 좌표
        wp_size_y = max(ys) - min(ys)
        print("max xs:", max(xs), "min xs:", min(xs))
        print("max ys:", max(ys), "min ys:", min(ys))
        print("wp_size_x:", wp_size_x, "wp_size_y:", wp_size_y)

        return wp_size_x, wp_size_y

def get_workpiece_surface_height(workpiece, surface_offset=0.0):
    """
    Workpiece 표면의 월드 Z 좌표 계산
    """
    vertices = get_workpiece_vertices(workpiece)
    if vertices is None:
        print(f"[get_surface_height] Failed to determine Z height, using fallback height 0.0955")
        return 0.0955
    else:
        zs = [v[2] for v in vertices]
        workpiece_z_size = max(zs) - min(zs)
        target_height = workpiece_z_size + surface_offset
        return target_height

def get_ee_pose(env: "ManagerBasedRLEnv", asset_name: str = "robot", ee_frame_name=EE_FRAME_NAME):
    """
    Returns end-effector pose (x, y, z, roll, pitch, yaw)
    -----------------------------------------------------
    - 현재 로봇의 joint 상태(q1~q6)를 불러와서
      FKSolver를 이용해 FK 계산 수행
    - FK 결과는 torch.Tensor (num_envs, 6) 형태로 반환
    """
    robot = env.scene[asset_name]
    q = robot.data.joint_pos[:, :6]  # (num_envs, 6)

    # 로봇 베이스 프레임 기준 EE 위치 및 자세 계산
    fk_solver = FKSolver(tool_z=0.239, use_degrees=False)
    ee_pose_local_list = []

    for i in range(env.num_envs):
        q_np = q[i].cpu().numpy().astype(float)
        ok, pose = fk_solver.compute(q_np, as_degrees=False)
        if not ok:
            ee_pose_local_list.append([float('nan')]*6)
        else:
            ee_pose_local_list.append([pose.x, pose.y, pose.z, pose.r, pose.p, pose.yaw])

    ee_pose_local = torch.tensor(ee_pose_local_list, dtype=torch.float32, device=q.device)
    ee_pos_local = ee_pose_local[:, :3]
    ee_rpy_local = ee_pose_local[:, 3:]

    ee_quat_local = quat_from_euler_xyz(ee_rpy_local[:, 0], ee_rpy_local[:, 1], ee_rpy_local[:, 2])

    # 로봇 베이스 프레임 -> 월드 프레임 변환
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    
    ee_pos_world = quat_apply(base_quat_w, ee_pos_local) + base_pos_w
    ee_quat_world = quat_mul(base_quat_w, ee_quat_local)

    roll_w, pitch_w, yaw_w = euler_xyz_from_quat(ee_quat_world)
    ee_rpy_world = torch.stack([roll_w, pitch_w, yaw_w], dim=1)

    ee_pose_world = torch.cat([ee_pos_world, ee_rpy_world], dim=1)
    # print(f"EE Position from FKSolver: {ee_pose_world[1].cpu().numpy()}")
    
    ee_index_lab = env.scene["robot"].body_names.index(ee_frame_name)
    ee_pos = env.scene["robot"].data.body_pos_w[:, ee_index_lab]
    ee_quat = robot.data.body_quat_w[:, ee_index_lab]
    roll, pitch, yaw = euler_xyz_from_quat(ee_quat)
    ee_rpy = torch.stack([roll, pitch, yaw], dim=1)

    ee_pose = torch.cat([ee_pos, ee_rpy], dim=1)

    return ee_pose

def ee_to_grid(env: "ManagerBasedRLEnv"):
    """
    EE 좌표를 grid 좌표로 변환
    """
    robot = env.scene["robot"]
    workpiece = env.scene["workpiece"]
    base_pos_w = robot.data.root_pos_w  # 로봇 베이스의 월드 위치 (Num_Envs, 3)
    WORKPIECE_REL_POS = torch.tensor([0.75, 0.0, 0.0], dtype=base_pos_w.dtype, device=env.device)
    wp_pos = base_pos_w + WORKPIECE_REL_POS.unsqueeze(0)

    ee_pos = get_ee_pose(env, asset_name="robot")[:, :3] # 위치 (x, y, z)만 사용

    if not hasattr(env, "wp_size_x"):
        wp_size_x, wp_size_y = get_workpiece_size(workpiece)
        env.wp_size_x = wp_size_x
        env.wp_size_y = wp_size_y
        wp_size_x_t = torch.tensor(env.wp_size_x, device=env.device)
        wp_size_y_t = torch.tensor(env.wp_size_y, device=env.device)
        env.grid_x_num = torch.ceil(wp_size_x_t / GRID_SIZE).long().item()
        env.grid_y_num = torch.ceil(wp_size_y_t / GRID_SIZE).long().item()
        
    wp_size_x_t = torch.tensor(env.wp_size_x, device=env.device)
    wp_size_y_t = torch.tensor(env.wp_size_y, device=env.device)
    grid_x_num = env.grid_x_num
    grid_y_num = env.grid_y_num
    
    origin_x = wp_pos[:, 0] - wp_size_x_t / 2
    origin_y = wp_pos[:, 1] - wp_size_y_t / 2

    ee_xy = ee_pos[:, :2].clone()
    ee_xy[:, 0] -= origin_x
    ee_xy[:, 1] -= origin_y

    grid_x = torch.clamp((ee_xy[:, 0] / GRID_SIZE).long(), 0, grid_x_num - 1)
    grid_y = torch.clamp((ee_xy[:, 1] / GRID_SIZE).long(), 0, grid_y_num - 1)

    # if env.episode_length_buf[0].item() < 2: # 에피소드 초기에만 출력
    #     print("\n--- GRID DEBUGGING INFO ---")
    #     # 로봇 베이스 월드 XY 좌표
    #     print(f"Robot Base World XY: {base_pos_w[ENV_ID, 0].item():.3f}, {base_pos_w[ENV_ID, 1].item():.3f}")
    #     # 작업물 중심 좌표
    #     print(f"Workpiece Center (wp_pos): {wp_pos[ENV_ID, 0].item():.3f}, {wp_pos[ENV_ID, 1].item():.3f}")
    #     # 그리드 원점 Origin 좌표
    #     print(f"Grid Origin XY: {origin_x[ENV_ID].item():.3f}, {origin_y[ENV_ID].item():.3f}")
    #     # EE의 월드 XY 좌표
    #     print(f"EE World XY (ee_pos): {ee_pos[ENV_ID, 0].item():.3f}, {ee_pos[ENV_ID, 1].item():.3f}")
    #     # 상대 EE 좌표 (Workpiece 원점 기준)
    #     print(f"EE Relative XY: {ee_xy[ENV_ID, 0].item():.3f}, {ee_xy[ENV_ID, 1].item():.3f}")
    #     # 계산된 그리드 인덱스
    #     print("---------------------------\n")
    # print(f"Calculated Grid Index: X={grid_x[ENV_ID]}, Y={grid_y[ENV_ID]}")

    return grid_x, grid_y

def reset_grid_mask(env: "ManagerBasedRLEnv", env_ids):
    """
    에피소드 리셋 시 env.grid_mask 텐서를 False(0)로 초기화하고,
    최초 호출 시 Grid Mask를 생성
    """
    workpiece = env.scene["workpiece"]

    # Grid 차원 정보가 env에 없으면 계산 및 저장
    if not hasattr(env, "wp_size_x"):
        wp_size_x, wp_size_y = get_workpiece_size(workpiece)
        env.wp_size_x = wp_size_x
        env.wp_size_y = wp_size_y
        wp_size_x_t = torch.tensor(env.wp_size_x, device=env.device)
        wp_size_y_t = torch.tensor(env.wp_size_y, device=env.device)
        env.grid_x_num = torch.ceil(wp_size_x_t / GRID_SIZE).long().item()
        env.grid_y_num = torch.ceil(wp_size_y_t / GRID_SIZE).long().item()

    # grid_mask 속성이 env에 없으면 초기 생성
    if not hasattr(env, "grid_mask"):
        env.grid_mask = torch.zeros((env.num_envs, env.grid_x_num, env.grid_y_num), dtype=torch.bool, device=env.device)

    # reset시 grid_mask 초기화
    env.grid_mask[env_ids] = False
    
    # Grid Mask 상태 관찰(obs)에서 사용하는 히스토리도 함께 초기화
    if hasattr(env, "_grid_mask_history"):
        env._grid_mask_history[env_ids] = 0.0

    # 리셋 시 누적 카운터 초기화
    if hasattr(env, "_revisit_count"):
        env._revisit_count[env_ids] = 0.0
        
    if hasattr(env, "_out_of_bounds_count"):
        env._out_of_bounds_count[env_ids] = 0.0

    return {}
