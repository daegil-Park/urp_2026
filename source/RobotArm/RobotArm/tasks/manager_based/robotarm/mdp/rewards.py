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
import RobotArm.tasks.manager_based.robotarm.mdp.path_manager as pm # 

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


# --- [핵심 보상 함수들] ---

def track_path_reward(env: ManagerBasedRLEnv):
    """
    [경로 추종] 목표 위치(Target)와 현재 위치(EE)가 가까울수록 높은 보상
    """
    # 1. 목표 위치 가져오기
    target_pos = _get_generated_target(env)
    
    # 2. 현재 로봇 끝단 위치 가져오기 (x, y, z)
    # Isaac Lab 표준: quat(4)가 뒤에 붙거나 앞에 붙음. 보통 state_w는 [pos(3), quat(4)]
    ee_state = get_ee_pose(env)
    current_pos = ee_state[:, :3]

    # 3. 거리 계산 (L2 Norm)
    distance = torch.norm(target_pos - current_pos, dim=-1)
    
    # 4. 점수 변환 (거리가 0이면 1점, 멀어지면 0점으로 수렴)
    # scale 파라미터(10.0)가 클수록 정밀한 추종을 요구함
    return torch.exp(-distance * 10.0)


def force_control_reward(env: ManagerBasedRLEnv):
    """
    [힘 제어] Z축으로 누르는 힘을 목표값(10N)에 맞추면 보상
    """
    target_force = 10.0 # 목표: 10 Newton
    
    if hasattr(env, "contact_forces"):
        # 센서 데이터에서 Z축 추출 (절대값 사용)
        current_force = torch.abs(env.contact_forces[:, 2])
    else:
        # 센서 없으면 0점 처리 (또는 에러 방지용 0)
        current_force = torch.zeros(env.num_envs, device=env.device)
        
    force_error = torch.abs(current_force - target_force)
    
    # 오차가 작을수록 큰 보상 (1.0 / (1 + 오차))
    # 오차 0N -> 1.0점, 오차 10N -> 약 0.1점
    return 1.0 / (1.0 + 0.2 * force_error)


def orientation_align_reward(env: ManagerBasedRLEnv):
    """
    [자세 유지] 툴이 바닥과 수직(Z축 정렬)을 유지하는지 평가
    """
    ee_state = get_ee_pose(env)
    ee_quat = ee_state[:, 3:7] # [qx, qy, qz, qw] (순서 확인 필요)
    
    # 목표 자세: 바닥을 향함 (아래쪽) -> 대략적인 벡터 내적 사용
    # 로봇 Tool의 Z축 벡터를 추출해서 (0, 0, -1)과 내적
    # (수학적으로 복잡하면, 간단히 r, p가 0인지 보는 것으로 대체 가능)
    
    # 여기서는 간단하게 'action'에서 회전 명령이 적을수록 좋다고 유도하거나
    # 현재 쿼터니언이 초기 자세와 비슷한지 비교하는 방식을 추천
    return 0.5 # (일단 기본 점수 부여, 추후 정밀 구현)


def action_smoothness_penalty(env: ManagerBasedRLEnv):
    """
    [부드러움] 로봇이 급격하게 움직이면(Jerk) 벌점(마이너스 점수)
    """
    # 이번 스텝의 행동(Action) 값의 크기가 클수록 감점 (에너지 최소화)
    # 또는 이전 행동과의 차이(Delta Action)를 사용
    if hasattr(env, "actions"):
        # Action 제곱의 합 (L2 Norm squared)
        return -torch.sum(torch.square(env.actions), dim=-1)
    return 0.0

# rewards_1.py 맨 아래에 추가

import carb.input # 키보드 입력을 받기 위한 라이브러리

def manual_termination(env: ManagerBasedRLEnv):
    """
    사용자가 'K' (Kill) 키를 누르면 모든 환경을 리셋(종료)시킵니다.
    """
    # 1. 입력 인터페이스 가져오기
    input_i = carb.input.acquire_input_interface()
    keyboard = input_i.get_keyboard()
    
    # 2. 'K' 키가 눌렸는지 확인 (눌리면 1.0 반환)
    # (원하시는 다른 키가 있다면 carb.input.KeyboardInput.SPACE 등으로 변경 가능)
    is_pressed = input_i.get_keyboard_value(keyboard, carb.input.KeyboardInput.K)
    
    # 3. 눌렸다면 모든 환경(num_envs)에 대해 True(종료) 신호 보냄
    if is_pressed > 0.5:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

# -----------------------------------------------
# [삭제/무시된 함수들]
# 기존의 new_visit_reward, revisit_penalty 등은 
# 폴리싱 경로 추종에는 방해가 되므로 이 파일에 포함하지 않았습니다.
# -----------------------------------------------