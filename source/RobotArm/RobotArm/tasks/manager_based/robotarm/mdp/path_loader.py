# path_loader.py
import torch
import numpy as np
import os

# [설정] 실제 파일 경로 (물결표 ~ 사용 가능)
FILE_PATH = os.path.expanduser("~/urp_ws1/tools/waypoints.txt")

# 데이터를 한 번 읽으면 저장해두는 변수 (캐싱)
_CACHED_PATH_TENSOR = None

def get_path_tensor(device="cpu"):
    global _CACHED_PATH_TENSOR

    # 1. 이미 읽은 데이터가 있으면 바로 반환 (속도 업)
    if _CACHED_PATH_TENSOR is not None:
        return _CACHED_PATH_TENSOR.to(device)

    # 2. 파일 읽기 시도
    if os.path.exists(FILE_PATH):
        try:
            print(f"[Info] 경로 파일 로딩 중: {FILE_PATH}")
            # txt 파일은 보통 공백 구분이므로 delimiter 없이 로드
            raw_data = np.loadtxt(FILE_PATH) 
            
            # 데이터가 1줄일 경우 예외 처리
            if raw_data.ndim == 1:
                raw_data = raw_data[np.newaxis, :]
            
            # (x, y, z) 좌표만 가져오기 (앞에서 3개)
            positions = raw_data[:, :3]
            
            _CACHED_PATH_TENSOR = torch.tensor(positions, dtype=torch.float32)
            print(f"[Success] 경로 로드 완료! 점 개수: {len(_CACHED_PATH_TENSOR)}")
            
        except Exception as e:
            print(f"[Error] 파일 읽기 실패: {e}")
            # 실패 시 비상용 직선 경로
            _CACHED_PATH_TENSOR = torch.tensor([[0.4, 0.0, 0.05], [0.5, 0.0, 0.05]], dtype=torch.float32)
    else:
        print(f"[Warning] 파일이 없습니다: {FILE_PATH}")
        print("기본 경로를 사용합니다.")
        _CACHED_PATH_TENSOR = torch.tensor([[0.4, -0.2, 0.05], [0.4, 0.2, 0.05]], dtype=torch.float32)

    return _CACHED_PATH_TENSOR.to(device)
