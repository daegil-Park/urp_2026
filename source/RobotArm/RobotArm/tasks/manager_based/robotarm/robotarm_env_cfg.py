# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import copy
import os

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

# Custom MDP modules
import isaaclab.envs.mdp as mdp
from .mdp import observations as local_obs 
from .mdp import rewards as local_rew      

from RobotArm.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# -------------------------------------------------------------------------
# Scene Configuration
# -------------------------------------------------------------------------

TEMP_ROBOT_CFG = copy.deepcopy(UR10E_W_SPINDLE_CFG)

# Contact Sensor Enable
TEMP_ROBOT_CFG.spawn = sim_utils.UsdFileCfg(
    usd_path=UR10E_W_SPINDLE_CFG.spawn.usd_path,
    activate_contact_sensors=True, 
    rigid_props=UR10E_W_SPINDLE_CFG.spawn.rigid_props,
    articulation_props=UR10E_W_SPINDLE_CFG.spawn.articulation_props,
)

@configclass
class RobotarmSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # 1. Ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # 2. Workpiece
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/eunseop/isaac/isaac_save/flat_surface_2.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 3. Robot
    robot: ArticulationCfg = TEMP_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        # [Safety] 바닥 충돌 방지 (Z=0.1)
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1), 
        ),
        # [Safety] 부드러운 제어를 위한 강성 조절
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=80.0,  
                damping=40.0,    
            ),
        }
    )

    # 4. Sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*", 
        history_length=3,
        track_air_time=False,
    )
    
    # 5. Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# -------------------------------------------------------------------------
# MDP Settings
# -------------------------------------------------------------------------

@configclass
class CommandsCfg:
    pass

@configclass
class ActionsCfg:
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.05, 
        use_default=True, 
    )
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        path_tracking = ObsTerm(func=local_obs.path_tracking_obs)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_history = ObsTerm(func=local_obs.ee_pose_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms."""
    
    # [기본] 경로 추종
    track_path = RewTerm(func=local_rew.track_path_reward, weight=10.0, params={"sigma": 0.1})
    
    # [핵심 수정] 수직 자세 가중치 극한 상향 (40.0)
    orientation = RewTerm(func=local_rew.orientation_align_reward, weight=40.0)
    
    # [신규] 자세 유지 (팔 꼬임 방지)
    joint_reg = RewTerm(func=local_rew.joint_deviation_reward, weight=5.0)
    
    # [기본] 힘 제어
    force_control = RewTerm(func=local_rew.force_control_reward, weight=2.0, params={"target_force": 10.0})

    # [Penalty]
    smoothness = RewTerm(func=local_rew.action_smoothness_penalty, weight=-0.1)
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-5.0)


@configclass
class EventCfg:
    # [Safety] 리셋 시 수직 자세(Drill Ready) 강제
    reset_robot_joints = EventTerm(
        func=local_rew.reset_robot_to_cobra, 
        mode="reset",
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 과도한 충돌 시 종료
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact, 
        params={
            "threshold": 1000.0, 
            "sensor_cfg": SceneEntityCfg("contact_forces") 
        }
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=128, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4            
        self.episode_length_s = 15.0   
        self.sim.dt = 1.0 / 120.0 
        self.sim.render_interval = self.decimation
        self.wp_size_x = 0.5
        self.wp_size_y = 0.5
