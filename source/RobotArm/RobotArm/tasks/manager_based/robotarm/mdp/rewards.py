# 파일 경로: RobotArm/tasks/manager_based/robotarm/mdp/rewards.py
from __future__ import annotations

import torch
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab.envs import ManagerBasedRLEnv

from . import path_loader

# -----------------------------------------------------------
# Global Logging
# -----------------------------------------------------------
_path_tracking_history = []
_force_control_history = []
_episode_counter = 0

# -----------------------------------------------------------
# Reward Functions
# -----------------------------------------------------------

def track_path_reward(env: ManagerBasedRLEnv, sigma: float = 0.1):
    """
    [경로 추종]
    목표 경로와의 거리(Euclidean Distance)를 최소화.
    """
    global _path_tracking_history, _episode_counter

    # 1. 경로 및 좌표계 설정
    path_tensor = path_loader.get_path_tensor(env.device)
    
    # [중요] 좌표계 오프셋
    # 로봇 베이스 기준(0,0)에서 작업물 위치(0.75, 0.0)로 경로를 이동
    # Z값 0.02는 가공 깊이를 고려한 보정값
    offset = torch.tensor([0.75, 0.0, 0.02], device=env.device)
    target_path_world = path_tensor + offset.unsqueeze(0)

    # 2. EE 위치
    robot = env.scene["robot"]
    current_pos = robot.data.body_pos_w[:, -1, :]

    # 3. 최소 거리 계산
    dists = torch.norm(current_pos.unsqueeze(1) - target_path_world.unsqueeze(0), dim=2)
    min_dist, _ = torch.min(dists, dim=1)

    # 4. 로깅
    step = int(env.common_step_counter)
    if env.num_envs > 0 and step % 10 == 0: # 과부하 방지: 10스텝마다 로깅
        _path_tracking_history.append((step, min_dist[0].item()))

    if hasattr(env, "max_episode_length"):
        if step > 0 and (step % int(env.max_episode_length) == int(env.max_episode_length) - 1):
            save_episode_plots(step)

    # [수정] 거리가 멀면 보상이 급격히 0이 되도록 sigma 조정
    return torch.exp(-torch.square(min_dist) / (sigma ** 2))


def orientation_align_reward(env: ManagerBasedRLEnv):
    """
    [자세 유지] - 논문 방식 적용 (Sharp Kernel)
    End-Effector가 바닥(-Z)을 수직으로 향하도록 강력하게 유도
    """
    robot = env.scene["robot"]
    ee_quat = robot.data.body_quat_w[:, -1, :]

    # Tool Z축 (Local +Z)
    tool_z = torch.zeros((env.num_envs, 3), device=env.device)
    tool_z[:, 2] = 1.0 
    tool_z_world = quat_apply(ee_quat, tool_z)
    
    # 목표 방향: World -Z
    target_dir = torch.zeros_like(tool_z_world)
    target_dir[:, 2] = -1.0 

    # Cosine Similarity
    dot_prod = torch.sum(tool_z_world * target_dir, dim=-1)
    
    # 오차: 0(완벽) ~ 2(반대)
    # dot_prod는 -1 ~ 1. 완벽 정렬 시 1.0
    error = 1.0 - dot_prod
    
    # [핵심] 오차가 조금이라도 있으면 보상을 확 깎음 (Sharpness: 10.0)
    return torch.exp(-error * 10.0)


def force_control_reward(env: ManagerBasedRLEnv, target_force: float = 10.0):
    """
    [힘 제어]
    접촉 시 너무 세지도, 약하지도 않은 힘(Target Force) 유지
    """
    global _force_control_history
    
    sensor = env.scene["contact_forces"]
    force_z = torch.abs(sensor.data.net_forces_w[..., 2])
    current_force, _ = torch.max(force_z, dim=-1) # (num_envs,)

    # 로깅
    if env.num_envs > 0 and env.common_step_counter % 10 == 0:
        _force_control_history.append((env.common_step_counter, current_force[0].item(), target_force))

    # [수정] 가우시안 커널 사용 (부드러운 보상)
    force_error = torch.abs(current_force - target_force)
    return torch.exp(-torch.square(force_error) / 50.0) # 분모가 클수록 관대함


def surface_approach_reward(env: ManagerBasedRLEnv, target_height: float = 0.05):
    """
    [신규 - 접근 보상]
    로봇이 허공에 있으면 힘 보상을 못 받으므로, 
    표면 근처(target_height)까지 내려오도록 유도하는 'Shaping Reward'.
    """
    robot = env.scene["robot"]
    ee_z = robot.data.body_pos_w[:, -1, 2]
    
    # 목표 높이와의 차이
    z_error = torch.abs(ee_z - target_height)
    
    # 5cm 이내로 들어오면 보너스, 아니면 페널티
    # 거리가 가까울수록 1.0에 가까워짐
    return torch.exp(-z_error * 5.0)


def joint_vel_penalty(env: ManagerBasedRLEnv):
    """[진동 방지] 속도 페널티 강화"""
    robot = env.scene["robot"]
    # 속도의 제곱합을 페널티로 부여
    return -torch.sum(torch.square(robot.data.joint_vel), dim=-1)


def action_smoothness_penalty(env: ManagerBasedRLEnv):
    """[떨림 방지] 이전 행동과의 차이(Action Derivative)를 최소화"""
    # Action Manager에서 현재 action을 가져옴
    action = env.action_manager.action
    
    # 만약 이전 액션을 저장할 수 있다면 좋겠지만, 
    # 여기서는 간단히 action의 크기(Magnitude)를 줄여서 급발진을 막음
    return -torch.sum(torch.square(action), dim=-1)


def out_of_bounds_penalty(env: ManagerBasedRLEnv):
    """[이탈 방지]"""
    robot = env.scene["robot"]
    ee_pos = robot.data.body_pos_w[:, -1, :]
    
    # 작업 영역 박스 정의
    is_out = (ee_pos[:, 0] < -0.2) | (ee_pos[:, 0] > 1.2) | \
             (ee_pos[:, 1] < -0.8) | (ee_pos[:, 1] > 0.8) | \
             (ee_pos[:, 2] < 0.0)  | (ee_pos[:, 2] > 0.8)
             
    return -1.0 * is_out.float()


# -----------------------------------------------------------
# Visualization Logic (수정 없음)
# -----------------------------------------------------------
def save_episode_plots(step: int):
    global _path_tracking_history, _force_control_history, _episode_counter
    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    os.makedirs(save_dir, exist_ok=True)

    if _path_tracking_history:
        steps, dists = zip(*_path_tracking_history)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, dists, label="Path Error", color="blue")
        plt.title(f"Episode {_episode_counter}")
        plt.savefig(os.path.join(save_dir, f"path_ep{_episode_counter}.png"))
        plt.close()

    _path_tracking_history.clear()
    _force_control_history.clear()
    _episode_counter += 1
