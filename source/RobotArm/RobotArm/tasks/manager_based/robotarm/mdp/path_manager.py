import torch
import numpy as np
import os
import csv

# 전역 변수로 경로 데이터를 저장 (메모리 효율 및 공유 위해)
GLOBAL_PATH_TENSOR = None
TOTAL_WAYPOINTS = 0

def load_csv_path(file_path: str, device: str = "cpu"):
    """
    CSV 파일을 읽어서 Tensor로 변환합니다.
    CSV 형식 가정: x, y, z, (옵션: nx, ny, nz)
    """
    global GLOBAL_PATH_TENSOR, TOTAL_WAYPOINTS
    
    if not os.path.exists(file_path):
        print(f"[Warning] CSV Path file not found at: {file_path}")
        print("Creating a dummy circular path instead...")
        # 파일이 없으면 임시로 원형 경로 생성 (에러 방지)
        t = np.linspace(0, 2*np.pi, 1000)
        x = 0.5 + 0.1 * np.cos(t)
        y = -0.3 + 0.1 * np.sin(t)
        z = np.full_like(t, 0.05)
        data = np.stack([x, y, z], axis=1)
    else:
        # CSV 읽기 (헤더가 있다면 skiprows=1 등 조정 필요)
        # 여기서는 헤더 없이 숫자만 있다고 가정
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
            # 만약 x,y,z만 필요하면 슬라이싱
            data = data[:, :3] 
        except Exception as e:
            print(f"[Error] Failed to load CSV: {e}")
            return None

    # 데이터를 텐서로 변환하여 GPU/CPU 메모리에 올림
    GLOBAL_PATH_TENSOR = torch.tensor(data, device=device, dtype=torch.float32)
    TOTAL_WAYPOINTS = len(data)
    print(f"[INFO] Path loaded! Total waypoints: {TOTAL_WAYPOINTS}")
    return GLOBAL_PATH_TENSOR


def get_target_pose_from_path(env):
    """
    현재 에피소드 시간(진행률)에 맞춰 목표 좌표를 반환합니다.
    """
    global GLOBAL_PATH_TENSOR, TOTAL_WAYPOINTS
    
    # 아직 로드가 안 됐다면 로드 시도 (경로 수정 필요!)
    if GLOBAL_PATH_TENSOR is None:
        # [주의] 실제 CSV 파일 경로로 수정하세요!
        # 예: "/home/user/RobotArm/path_data/zigzag_path.csv"
        csv_path = "/tmp/sample_path.csv" 
        load_csv_path(csv_path, device=env.device)
        
    device = env.device
    num_envs = env.num_envs
    
    # 1. 현재 진행 상황 계산
    # 에피소드 시간(초)에 따라 인덱스 결정
    # 속도 제어: 1초에 몇 개의 점을 지날 것인가? (예: 50포인트/초)
    speed_factor = 50.0 
    
    current_time = env.episode_length_buf * env.step_dt # 현재 시간 (초)
    
    # 인덱스 계산 (float -> long)
    # clamp로 인덱스가 배열 범위를 넘지 않게 막음
    indices = (current_time * speed_factor).long()
    indices = torch.clamp(indices, 0, TOTAL_WAYPOINTS - 1)
    
    # 2. 각 환경(Env)별 목표점 추출
    # 모든 로봇이 똑같은 경로를 시차 없이 따라간다고 가정 (동기화)
    # 만약 로봇마다 다르게 하려면 random offset을 줬어야 함.
    
    # GLOBAL_PATH_TENSOR: [Total_Points, 3]
    # indices: [Num_Envs]
    
    # gather를 쓰거나 인덱싱 사용
    target_pos = GLOBAL_PATH_TENSOR[indices] # [Num_Envs, 3]
    
    return target_pos