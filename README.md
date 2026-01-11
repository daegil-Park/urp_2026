# RobotArm - Isaac Lab Project

본 프로젝트는 **Isaac Lab** 기반의 강화학습 환경으로,
UR10e 로봇 암이 workpiece와 상호작용하며 작업을 수행하도록 학습하는 것을 목표로 합니다.
학습 및 평가에는 `skrl` 강화학습 프레임워크를 사용합니다.
추가 - 기존 코드 및 파일 중 일부 수정된 것이 있음. 기존 2025 urp 관련 파일 중 수정이 된 것은 기존 파일 유지를 위해 이름 뒤에 old를 넣었고 새로 다시 만든 코드에 기존 이름을 덧붙여 사용
추가 - old가 있는 파일이 없는 것은 따로 건드리지 않은 것임. 또한 path_mamager.py의 경우 아예 새로 생성한 코드.
<br />
<br />

---
<br />

## 학습 및 실행 (Train & Play)
### 학습 (Train)
```bash
~/IsaacLab/isaaclab.sh -p ~/RobotArm2026/scripts/skrl/train.py --task Template-Robotarm-v0
```

### 실행 (Play)
```bash
~/IsaacLab/isaaclab.sh -p ~/RobotArm2026/scripts/skrl/play.py --task Template-Robotarm-v0
```
<br />

## 로봇 모델 설정 (Robot Model Configuration)
UR10e 로봇 모델은 USD 파일로 로드됩니다.

### 파일 경로
```bash
RobotArm/source/RobotArm/RobotArm/robots/ur10e_w_spindle.py
```

### 코드
```python
UR10E_USD_PATH = "/home/eunseop/isaac/isaac_save/ur10e_tuning2.usd"
```
다른 로봇 모델을 불러오려면 **UR10E_USD_PATH** 변경.
<br />
<br />
## Workpiece 설정 (Workpiece Configuration)
Workpiece는 환경 설정 파일에서 USD asset으로 정의됩니다.

### 파일 경로
```bash
RobotArm/source/RobotArm/RobotArm/robots/ur10e_w_spindle.py
```

### 코드
```python
workpiece = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Workpiece",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/eunseop/isaac/isaac_save/flat_surface_2.usd"
    ),
)
```
다른 workpiece 모델을 불러오려면 **usd_path** 변경.
<br />
<br />
## Reward 계산을 위한 Mesh 경로 변경
reward.py 파일 내에서 workpiece를 불러오기 위해서는 **mesh prim path**가 사용중인 USD 파일 내의 구조와 동일해야 합니다.

### 파일 경로
```bash
source/RobotArm/RobotArm/tasks/manager_based/robotarm/mdp/rewards.py
```

### 코드
```python
mesh_prim = (
    workpiece_prim
    .GetChild("World")
    .GetChild("flat_surface_5")
    .GetChild("mesh_")
    .GetChild("Mesh")
)
```

### Mesh 경로 확인 방법 (Isaac Sim)

1. Isaac Sim에서 workpiece USD 파일 열기
2. 화면 우측의 **Stage** 탭 열기
3. **Mesh**까지의 hierarchy 확인
4. rewards.py 내에서 **GetChild()**를 이용하여 동일한 구조 반영

예시:
```markdown
World
 └── flat_surface_5
     └── mesh_
         └── Mesh
```
<br />

## 주의사항
- 로봇 USD 변경 시 모델 경로 수정
- Workpiece USD 변경 시: usd_path 수정, rewards.py 내 mesh prim 경로 수정
- 잘못된 mesh 경로는 reward 계산 오류를 유발할 수 있음
<br />

---
<br />

## 보상 및 유틸리티 함수
본 섹션에서는 reward.py 파일 내에 정의된 보상 함수, 종료 조건, 로깅 및 각종 유틸리티 함수들을 설명합니다.
<br />
<br />
### 보상 함수 (Reward Functions)

`coverage_reward(env, exp_scale=5.0)`
- 엔드이펙터가 방문한 **grid cell coverage 증가량**을 기반으로 보상 계산
- workpiece 영역을 grid로 분할하고 방문 여부를 `grid_mask`로 관리
- 전체 coverage 비율에 **지수 스케일링(exponential scaling)** 적용
- coverage 진행 상황을 CSV 파일로 로깅


`ee_movement_reward(env, max_movement=0.05)`
- 엔드이펙터의 이동 거리에 비례한 보상
- 로봇이 정지한 상태에 머무르는 정책을 방지
- 과도한 움직임 보상을 방지하기 위해 이동 거리를 클리핑

`out_of_bounds_penalty(env)`
- 엔드이펙터가 workpiece의 XY 범위를 벗어났을 경우 패널티 부여
- 이탈 횟수를 누적하여 **반복 이탈 시 패널티가 점점 증가**
- 작업 영역 내에서의 안정적인 탐색을 유도

`revisit_penalty(env)`
- 이전 step 대비 coverage가 증가하지 않았을 경우 패널티 부여
- 이미 방문한 grid cell을 반복 방문하는 행동 억제
- 중복 방문 횟수에 따라 패널티가 누적 증가

`surface_proximity_reward(env)`
- 엔드이펙터의 Z 위치가 workpiece 표면 높이에 가까울수록 높은 보상
- 일정 오차 범위(5cm)는 허용하여 안정적인 표면 추종 유도

`ee_orientation_alignment(env, target_axis=(0,0,-1))`
- 엔드이펙터의 Z축 방향이 목표 축과 정렬되도록 유도
- 내적(dot product)을 사용해 정렬 정도를 계산
- EE 자세 정렬을 통한 작업 안정성 확보

`time_efficiency_reward(env, max_steps=1000)`
- 에피소드 진행 시간이 짧을수록 높은 보상
- 빠른 task 수행을 유도하는 시간 효율성 보상

`distance_to_workpiece_reward(env)`
- 엔드이펙터가 workpiece 중심에 가까울수록 높은 보상
- 작업 영역 가장자리로 벗어나는 것을 방지하는 자석 형태의 보상
<br />

### 종료 조건 (Termination Conditions)
`check_coverage_success(env)`
- 전체 workpiece grid 중 **95% 이상 커버 시 성공으로 판단**
- 성공 여부를 boolean tensor로 반환하여 episode 종료 조건에 사용
<br />

### 로그 저장용 함수

`init_coverage_logger()`
- coverage 기록을 저장할 CSV 파일 초기화
- 기존 파일이 있으면 삭제 후 헤더 작성

`log_coverage_data(env, coverage_ratio)`
- 특정 환경의 coverage 비율과 step 정보를 CSV에 기록
- 학습 중 coverage 진행 상황 분석용
<br />

### 기타 유틸리티 함수

`get_workpiece_vertices(workpiece)`
- USD Mesh에서 workpiece의 vertex 좌표를 추출
- USD 내부 prim 구조에 따라 mesh 경로 수정 필요

`get_workpiece_size(workpiece)`
- workpiece의 X, Y 크기를 vertex 좌표 기반으로 계산
- 실패 시 기본 크기(0.5 x 0.5) 사용

`get_workpiece_surface_height(workpiece, surface_offset=0.0)`
- workpiece 표면의 월드 Z 좌표 계산
- surface offset을 적용해 EE 목표 높이 설정 가능

`get_ee_pose(env, asset_name="robot")`
- 로봇 joint 상태를 기반으로 FK 계산 수행
- 엔드이펙터의 월드 좌표계 기준
  `(x, y, z, roll, pitch, yaw)` 반환
- FKSolver와 Isaac Lab body state를 함께 사용

`ee_to_grid(env)`
- 엔드이펙터의 월드 좌표를 workpiece 기준 grid 좌표로 변환
- coverage 계산을 위한 grid index `(grid_x, grid_y)` 반환

`reset_grid_mask(env, env_ids)`
- 에피소드 리셋 시 grid_mask 및 관련 누적 변수 초기화
- coverage, revisit, out-of-bounds 카운터 리셋
<br />

---
<br />

## 관찰 함수 (Observation Functions)
본 섹션에서는 observation.py 파일 내에 정의된 관찰 함수들을 설명합니다.

`grid_mask_state_obs(env, grid_mask_history_len=4)`
- Workpiece 영역을 grid로 나눈 뒤 **방문 상태(grid mask)** 를 관찰값으로 제공
- 각 grid cell은 방문 여부(True / False)를 나타냄
- 최근 `grid_mask_history_len` step 동안의 grid 상태를 버퍼에 저장, 관찰
- 에피소드 리셋 시 히스토리를 0으로 초기화

`ee_pose_history(env, history_len=5)`
- 엔드이펙터(EE)의 위치 및 자세 히스토리를 관찰값으로 제공
- 각 step에서 EE pose는 다음 형식으로 저장됨: [x, y, z, roll, pitch, yaw]
- 최근 `history_len` step 동안의 EE pose를 버퍼에 저장, 관찰
- 시간적 움직임 패턴(속도, 방향성)을 학습할 수 있도록 지원
