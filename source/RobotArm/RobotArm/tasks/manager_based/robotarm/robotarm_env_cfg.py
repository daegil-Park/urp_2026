# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as TermTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

# --- [수정됨] 커스텀 액션 클래스를 위해 필요한 모듈 ---
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv

# Pre-defined configs (프랑카 로봇 사용 예시)
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG

##
# Custom Action Class (에러 수정 버전)
##

class HybridPolishingAction(ActionTerm):
    """
    엔드 이펙터(EE) 제어를 위한 하이브리드 액션 클래스.
    (에러를 유발하던 _debug_draw 기능을 제거하고 단순화한 버전)
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        # 상위 클래스 초기화
        super().__init__(cfg, env)
        
        # 로봇 객체 가져오기
        self.robot = env.scene[cfg.asset_name]
        
        # 관절 이름으로 인덱스 찾기 (기본값: 모든 관절)
        self.joint_indices = self.robot.find_joints(".*")[0]
        
        # 액션 차원 설정 (예: 6개 관절 + 1개 힘 = 7)
        # 만약 에러가 나면 이 숫자를 로봇의 실제 관절 수(예: 7 또는 9)로 맞추세요.
        self.action_dim = len(self.joint_indices) 
        
        # 디버깅용 목표 높이
        self.target_z = 0.8

    def process_actions(self, actions: torch.Tensor):
        """
        모델이 출력한 액션을 시뮬레이션에 적용할 형태로 변환합니다.
        """
        # 1. 액션 스케일링 (너무 빠르게 움직이지 않도록 0.1 곱함)
        scaled_actions = actions * 0.1
        
        # 2. 현재 관절 위치 가져오기
        current_joint_pos = self.robot.data.joint_pos[:, self.joint_indices]
        
        # 3. 목표 관절 위치 계산 (현재 위치 + 액션)
        # 복잡한 IK 대신 일단 관절을 직접 움직이게 하여 에러를 방지합니다.
        target_joint_pos = current_joint_pos + scaled_actions
        
        # 4. 관절 한계(Limit)를 넘지 않도록 클램핑
        # (필요시 추가 구현, 지금은 생략)
        
        return target_joint_pos

    def apply_actions(self, actions: torch.Tensor):
        """
        변환된 액션을 실제로 로봇에게 명령합니다.
        """
        target_pos = self.process_actions(actions)
        self.robot.set_joint_position_target(target_pos, joint_ids=self.joint_indices)


##
# Scene Configuration
##

@configclass
class RobotArmSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robot and a table."""

    # 1. Robot (Franka Panda)
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 2. Table (Simple Cube)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.4), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.2)),
    )

    # 3. Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP Settings (Actions, Obs, Rewards)
##

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    # 위에서 정의한 '안전한' 커스텀 액션 사용
    arm_action = ActionTermCfg(
        asset_name="robot",
        body_name="panda_hand", # 로봇 손 이름
        class_type=HybridPolishingAction,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        # 로봇의 관절 위치와 속도를 관측
        joint_pos = ObsTerm(func=lambda env: env.scene["robot"].data.joint_pos)
        joint_vel = ObsTerm(func=lambda env: env.scene["robot"].data.joint_vel)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # 시작할 때 로봇 상태 리셋
    reset_robot = EventTerm(
        func=lambda env, env_ids: env.scene["robot"].reset(env_ids),
        mode="reset",
    )

@configclass
class RewardsCfg:
    """Reward terms for the environment."""
    # 살아있는 것만으로도 점수 (테스트용)
    alive = RewTerm(func=lambda env: 1.0, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the environment."""
    # 500스텝 지나면 에피소드 종료
    time_out = TermTerm(func=lambda env: env.episode_length_buf > 500, time_out=True)


##
# Environment Configuration
##

@configclass
class RobotArmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the RobotArm environment."""
    # Scene settings
    scene: RobotArmSceneCfg = RobotArmSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Physics settings (기본값)
    def __post_init__(self):
        self.sim.dt = 0.01  # 시뮬레이션 스텝 시간
        self.sim.render_interval = 4  # 렌더링 간격
