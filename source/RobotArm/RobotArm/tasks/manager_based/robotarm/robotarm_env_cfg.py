# ... (이전 import 문들은 그대로 유지) ...

# =========================================================================
# [Action] 수직 하강 (Drill Press Style) & 하이브리드 제어
# =========================================================================
class HybridPolishingAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = env.scene.find_joints(cfg.asset_name, cfg.joint_names)
        self.num_joints = len(self.joint_ids)
        self._action_dim = 12 
        
        # State Machine
        # 0: Align (상공에서 위치 잡기)
        # 1: Descend (수직으로만 내려가기)
        # 2: Polishing (RL 개입 시작)
        self.state = torch.zeros(env.num_envs, dtype=torch.int, device=env.device) 
        self.timer = torch.zeros(env.num_envs, device=env.device)
        self.contact_z = torch.zeros(env.num_envs, device=env.device)

        # RL 파라미터 범위
        self.k_pos_range = torch.tensor([10.0, 1500.0], device=env.device)
        self.k_rot_range = torch.tensor([100.0, 1000.0], device=env.device) 
        self.d_range = torch.tensor([10.0, 150.0], device=env.device)

        # 작업 중심점
        self.center_x = 0.75
        self.center_y = 0.0
        self.path_timer = torch.zeros(env.num_envs, device=env.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt
        self.timer += dt
        
        # 1. 로봇 상태 가져오기
        robot = self._env.scene[self.cfg.asset_name]
        ee_pos = robot.data.body_pos_w[:, -1, :]   # [num_envs, 3]
        ee_quat = robot.data.body_quat_w[:, -1, :] # [num_envs, 4]
        
        # 접촉 센서 (Z축 힘)
        sensor = self._env.scene.sensors["contact_forces"]
        force_z = torch.abs(sensor.data.net_forces_w[..., 2]).max(dim=-1)[0]

        # 2. 목표값 초기화
        target_pos = ee_pos.clone()
        # [핵심] 무조건 바닥을 보는 수직 쿼터니언 고정 (변하지 않음)
        # (UR10e 기준: Rotate X 180 deg -> [0, 1, 0, 0])
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self._env.device).repeat(self._env.num_envs, 1)

        # RL 파라미터 디코딩
        k_pos = scale_transform(actions[:, 0:3], self.k_pos_range[0], self.k_pos_range[1])
        k_rot = scale_transform(actions[:, 3:6], self.k_rot_range[0], self.k_rot_range[1])
        d_pos = scale_transform(actions[:, 6:9], self.d_range[0], self.d_range[1])
        d_rot = scale_transform(actions[:, 9:12], self.d_range[0], self.d_range[1])

        # ==================================================================
        # [State 0] Align: 상공 정렬 (XY 이동, Z 고정)
        # ==================================================================
        mask_align = (self.state == 0)
        if torch.any(mask_align):
            # 목표: 작업점 상공 20cm
            target_pos[mask_align, 0] = self.center_x
            target_pos[mask_align, 1] = self.center_y
            target_pos[mask_align, 2] = 0.20 
            
            # 강성 최대 (빠르고 정확하게 이동)
            k_pos[mask_align] = 2000.0
            k_rot[mask_align] = 1000.0 
            d_pos[mask_align] = 100.0

            # XY 위치가 맞고, 높이도 얼추 맞으면 다음 단계로
            err = torch.norm(target_pos[mask_align] - ee_pos[mask_align], dim=-1)
            ready = (self.timer > 1.5) & (err < 0.02)
            
            switch_ids = torch.nonzero(mask_align).flatten()[ready]
            self.state[switch_ids] = 1
            self.timer[switch_ids] = 0.0
        
        # ==================================================================
        # [State 1] Descend: 수직 하강 (XY 고정, Z 감소)
        # ==================================================================
        mask_descend = (self.state == 1)
        if torch.any(mask_descend):
            # [핵심] XY는 절대 움직이지 않음 (수직 유지)
            target_pos[mask_descend, 0] = self.center_x
            target_pos[mask_descend, 1] = self.center_y
            
            # Z축만 현재 위치보다 아주 조금 아래로 설정 (속도 제어 효과)
            # 0.5mm 씩 내림 -> 부드러운 하강
            target_pos[mask_descend, 2] = ee_pos[mask_descend, 2] - 0.0005
            
            # 강성 조절 (충격 흡수 대비)
            k_pos[mask_descend] = 800.0
            k_rot[mask_descend] = 1000.0 # 회전은 여전히 꽉 잡음 (기울어짐 방지)
            
            # 접촉 감지 (2N 이상 닿으면 멈춤)
            contacted = (force_z > 2.0)
            switch_ids = torch.nonzero(mask_descend).flatten()[contacted[mask_descend]]
            
            if len(switch_ids) > 0:
                self.state[switch_ids] = 2
                self.contact_z[switch_ids] = ee_pos[switch_ids, 2] # 닿은 높이 저장
                self.timer[switch_ids] = 0.0
                self.path_timer[switch_ids] = 0.0

        # ==================================================================
        # [State 2] Polishing: 작업 시작 (RL + ㄹ자 패턴)
        # ==================================================================
        mask_polish = (self.state == 2)
        if torch.any(mask_polish):
            self.path_timer[mask_polish] += dt
            t = self.path_timer[mask_polish]
            
            # 저장된 바닥 높이(contact_z)보다 2mm 더 누름 (10N 생성)
            target_z = self.contact_z[mask_polish] - 0.002
            
            # 이제서야 비로소 XY가 움직임 (ㄹ자 패턴)
            path_x = self.center_x + 0.15 * torch.sin(0.2 * t)
            path_y = 0.2 * torch.sin(3.0 * t)
            
            target_pos[mask_polish, 0] = path_x
            target_pos[mask_polish, 1] = path_y
            target_pos[mask_polish, 2] = target_z
            
            # RL이 출력한 강성 적용 (단, 회전 강성 최소값은 보장)
            k_rot[mask_polish] = torch.clamp(k_rot[mask_polish], min=300.0)


        # 3. 토크 계산 (OSC)
        pos_err = target_pos - ee_pos
        
        quat_inv = quat_conjugate(ee_quat)
        q_diff = quat_mul(target_quat, quat_inv)
        rot_err = 2.0 * torch.sign(q_diff[:, 0]).unsqueeze(1) * q_diff[:, 1:]
        
        vel_lin = robot.data.body_vel_w[:, -1, :3]
        vel_ang = robot.data.body_vel_w[:, -1, 3:]
        
        F_pos = k_pos * pos_err - d_pos * vel_lin
        F_rot = k_rot * rot_err - d_rot * vel_ang
        
        F_task = torch.cat([F_pos, F_rot], dim=-1)
        
        jacobian = robot.data.jacobian_w[:, self.joint_ids, :]
        j_t = jacobian.transpose(-2, -1)
        desired_torque = torch.bmm(j_t, F_task.unsqueeze(-1)).squeeze(-1)
        
        robot.set_joint_effort_target(desired_torque, joint_ids=self.joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = 0
        self.timer[env_ids] = 0.0
        self.path_timer[env_ids] = 0.0

# =========================================================================
# Scene Config (물리 안정성 최우선)
# =========================================================================
USER_STL_PATH = "/home/nrs2/RobotArm2026/flat_surface.stl"
# 수직 자세 (Ready Pose)
DEVICE_READY_STATE = {
    "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.5708, "elbow_joint": -1.5708,
    "wrist_1_joint": -1.5708, "wrist_2_joint": 1.5708, "wrist_3_joint": 0.0,
}

@configclass
class RobotarmSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # [핵심] 초록색 박스 (단단한 바닥, STL 대체)
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.1), # 1m x 1m 바닥
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16, # 물리 계산 4배 강화 (뚫림 방지)
                solver_velocity_iteration_count=8,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, # 5mm 전부터 충돌 감지
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, 0.0)),
    )

    # [핵심] 로봇 설정 (힘 제한)
    robot: ArticulationCfg = UR10E_W_SPINDLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=DEVICE_READY_STATE),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"], 
                effort_limit=150.0, # 150Nm로 제한 (뚫고 들어가는 힘 억제)
                velocity_limit=100.0,
                stiffness=0.0, # Torque Mode
                damping=2.0,   # 최소한의 공기 저항
            ),
        }
    )
    # 로봇 자체 충돌 설정 강화
    robot.spawn.rigid_props.enable_ccd = True 
    robot.spawn.rigid_props.solver_position_iteration_count = 16

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ur10e_w_spindle_robot/.*", history_length=3, track_air_time=False,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# =========================================================================
# MDP & Rewards
# =========================================================================
@configclass
class ActionsCfg:
    # 하이브리드 액션 연결
    polishing = ActionTerm(func=HybridPolishingAction, params={"asset_name": "robot", "joint_names": [".*"]})
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
    # 1. Force Tracking (10N 유지) - 가장 중요
    force_tracking = RewTerm(func=local_rew.force_tracking_reward, weight=100.0, params={"target_force": 10.0})
    
    # 2. Orientation (수직 유지)
    orientation_align = RewTerm(func=local_rew.orientation_align_reward, weight=80.0, params={"target_axis": (0,0,-1)})
    
    # 3. Path Tracking (ㄹ자 경로)
    track_path = RewTerm(func=local_rew.track_path_reward, weight=30.0, params={"sigma": 0.1})
    
    # 4. Penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    out_of_bounds = RewTerm(func=local_rew.out_of_bounds_penalty, weight=-10.0)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 수직 자세가 45도 이상 무너지면 즉시 종료
    bad_orientation = DoneTerm(func=local_rew.bad_orientation_termination, params={"limit_angle": 0.78})

@configclass
class EventCfg:
    # 초기화 시 약간의 랜덤성 부여 (Robustness)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0)}
    )

@configclass
class RobotarmEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotarmSceneCfg = RobotarmSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        # 물리 안정화 설정
        self.sim.substeps = 2
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.enable_stabilization = True
