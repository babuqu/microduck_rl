"""Microduck velocity (walking) environment.

The main locomotion task: velocity-command tracking + head-pose commands.
The reward/regularization recipe is locomotion-focused (lean tracking +
gait/feet terms, curriculum-ramped action-rate smoothing), with:

  - foot_slip kept at -0.1 (deliberately weak — stronger was too restrictive
    for this robot's pivot-heavy turning)
  - fixed, modest command ranges (ang ±1.0 makes turning learnable) instead of
    a widening curriculum that outpaced the robot's capability
  - turn-in-place: 15% of envs get lin=0 + |ang| ∈ [0.4, 1.0] (2026-07 audit:
    independent uniform sampling makes spin-on-the-spot ~2% of data → untrained)
  - head_pose_tracking as a primary objective, plus an EMA-based head_pose_bias
    penalty that prices only the escapable DC head droop (see below)
  - body_pose tracking infra kept intact but DISABLED (weight 0) so the obs
    slot stays alive for envs that use it

中文说明：
Microduck 速度（行走）环境——主 locomotion 任务：速度指令跟踪 + 头部位姿指令。
奖励/正则化配方以运动为中心（倾斜跟踪 + 步态/足端项，action_rate 平滑按
课程渐进加权），要点：

  - foot_slip 保持 -0.1（刻意偏弱——更强会限制这台机器人以枢轴为主的转向）
  - 固定、适中的指令范围（角速度 ±1.0 让转向可学），不用超出机器人能力的
    拓宽式课程
  - 原地旋转：15% 环境被指令 lin=0 + |ang| ∈ [0.4, 1.0]（2026-07 审计：
    独立均匀采样使原地旋转仅占约 2% 数据 → 训练不到）
  - head_pose_tracking 为主要目标，另加基于 EMA 的 head_pose_bias 惩罚，
    只对"可逃脱的"头部 DC 下垂计价（见下）
  - body_pose 跟踪基础设施保留但禁用（权重 0），保持 obs 槽位存活，
    供需要它的环境使用
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24
# 被指令原地旋转的环境比例（线速度=0，角速度绝对值 ∈ [0.4·最大值, 最大值]）。
# Fraction of envs commanded to spin on the spot (lin=0, |ang| ∈ [0.4·max, max]).
TURN_IN_PLACE_FRACTION = 0.15

# Symmetry 对称性（镜像损失，默认关闭）
ENABLE_SYMMETRY = False

# 领域随机化开关 / Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True  # 随机化躯干（trunk_base）质心位置
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies / 随机化头部组件各刚体的质心
ENABLE_KP_RANDOMIZATION = False # Was True / 电机 KP 增益随机化（曾为 True）
ENABLE_KD_RANDOMIZATION = False # Was True / 电机 KD 增益随机化（曾为 True）
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable / 质量+惯量随机化（行走稳定后可开启）
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale / 通过 friction_scale 按环境缩放 BAM 的摩擦预算
ENABLE_JOINT_DAMPING_RANDOMIZATION = False  # 关节阻尼随机化
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed). / 转子反射惯量（microban 风格）。对 BAM 生效（armature 是被设置而非清零）。
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training / 基于速度的推扰，用于鲁棒性训练
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors / 模拟 IMU 安装误差
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias) / 每环境恒定的关节编码器标定偏移（actor 观测读 joint_pos + bias）
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior / 随机化初始倾斜，迫使策略学会响应

# 头部/躯干位姿指令跟踪（取代旧的颈部偏置扰动方案）。
# 头部位姿：neck/head 各关节相对 HOME 的 4 维增量；速度环境将其作为主要跟踪
# 目标。躯干位姿：[x, y, z, roll, pitch, yaw] 的 6 维增量；速度环境只采样小
# 范围 + 极小奖励权重，让输入神经元保持活性但不主导策略（standup 环境再调高）。
# Head/body pose command tracking (replaces the old neck-offset disturbance scheme).
# Head pose: 4D deltas-from-HOME on neck/head joints; vel env tracks these as a
# primary objective. Body pose: 6D delta in [x, y, z, roll, pitch, yaw]; vel env
# samples small ranges + tiny reward weight so input neurons stay alive but
# tracking isn't the priority (standup env raises the weight).
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# 观测配置 / Observation configuration
USE_PROJECTED_GRAVITY = True  # True 时用投影重力代替原始加速度计读数

# 领域随机化范围（按需调整）/ Domain randomization ranges (adjust as needed)
# 保守且已被验证稳定的范围——需要时可逐步加大
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # 初始 ±3mm，经课程渐进至 ±8mm (±3mm initial, ramped to ±8mm via curriculum)
# 头部质心随机化：每个 episode 对头部组件的每个刚体重新采样
# （neck → neck_pitch → yaw_roll_motion → head-roll 刚体）。与上面躯干质心
# 随机化相同的非累积机制。head-roll 刚体在 walk 模型名为 bottom_head_shell、
# 在 2026-07 roller 模型名为 jaw_soft，故用交替正则。注意：bearing_roll 并非
# 头部刚体——在两个模型里它都是右髋 yaw 连杆（trunk_base 的子级）；它一直被
# 误列在此，仅为保持既有 DR 行为而保留。
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck → neck_pitch → yaw_roll_motion → head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body — in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # 初始 ±3mm，经课程渐进 (±3mm initial, ramped via curriculum)
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # 质量与惯量一同缩放 ±5% (±5% applied to BOTH mass and inertia together)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10%（可加大到 0.8-1.2）
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)  # 关节摩擦缩放范围
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)  # 关节阻尼缩放范围
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # 转子反射惯量 ±10%（microban：dr.joint_armature，同范围）
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # 每 3-6 秒施加一次推扰
# 速度改变范围（m/s）。曾为 ±0.5——每 3-6 秒一次、比最大步速（0.4）还大的加性
# 冲击，会训练出长期"神经质"的摔倒恢复步态（2026-07 审计）。±0.3 在保留抗推扰
# 鲁棒性的同时，让更平静的步态成为最优。
# Velocity change range in m/s. Was ±0.5 — an ADDITIVE kick larger than max walk
# speed (0.4) every 3-6 s trains a permanently nervous fall-recovery gait
# (2026-07 audit). ±0.3 keeps push robustness while letting a calmer gait be optimal.
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
# IMU 安装误差最高 6°（随机轴）。注意：零均值（随机轴）——训练的是对失准*幅度*
# 的容忍，不是俯仰偏置。真实主板系统性 ~5° 俯仰偏置在运行时源头修正
# （imu-pitch-offset），不在这里。
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # up-to-6° random-axis IMU mounting error. NOTE: zero-centered (random axis) — trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # 每关节编码器偏移 ±0.86°（每环境恒定）
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # episode 开始时前后倾斜 ±10°
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # episode 开始时左右倾斜 ±5°

import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Microduck 专用粗糙地形：比默认 ROUGH_TERRAINS_CFG 温和得多。
# 机器人抬脚只有 ~1-2 cm，台阶上限设为 1.5 cm。
# Microduck-specific rough terrain: much gentler than the default ROUGH_TERRAINS_CFG.
# The robot can only lift its feet ~1-2 cm, so steps are capped at 1.5 cm.
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),# 地块尺寸（m）
    border_width=20.0,# 边界宽度（m）
    num_rows=10,# 行数
    num_cols=20,# 列数
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),# 平面地形
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,# 台阶地形比例
            step_height_range=(0.0, 0.015),  # 台阶最高 1.5 cm（默认 10 cm）(max 1.5 cm (vs 10 cm default))
            step_width=0.15,# 台阶宽度（m）
            platform_width=2.0,# 平台宽度（m）
            border_width=1.0,# 边界宽度（m）
        ),
        # 注意：已移除 BoxInvertedPyramidStairsTerrainCfg——它把 env_origin_z 设为
        # 坑底（负值），导致重置时 root_z = 0.12 + env_origin_z ≈ −0.10 m，机器人
        # 出生在坑底之下并穿地。
        # NOTE: BoxInvertedPyramidStairsTerrainCfg removed — it sets env_origin_z to the pit
        # bottom (negative), causing resets at root_z = 0.12 + env_origin_z ≈ −0.10 m which
        # places the robot below the pit floor and makes it fall through the ground.
        # 起伏鹅卵石地面：每格随机高度偏移。
        # 8m 地块上 grid_width=0.12 → 66×66 = 4 356 盒/地块 → 总计 ~26 万 → OOM。
        # 0.45 m 给出 17×17 = 289 盒/地块 → 总计 ~1.7 万（border = 0.35 m ✓）。
        # 不能整除地形尺寸（8.0 m）：0.45 × 17 = 7.65 ✓
        # Uneven cobblestone-like ground: random per-cell height offsets.
        # grid_width=0.12 on an 8m patch = 66×66 = 4 356 boxes/patch → ~261 K total → OOM.
        # 0.45 m gives 17×17 = 289 boxes/patch → ~17 K total (border = 0.35 m ✓).
        # Must not divide evenly into terrain size (8.0 m): 0.45 × 17 = 7.65 ✓
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,# 随机网格地形比例
            grid_width=0.45,# 网格宽度（m）
            grid_height_range=(0.0, 0.010),  # 最高 1 cm (max 1 cm)
            platform_width=1.5,# 平台宽度（m）  
        ),
        # 缓坡（高度场金字塔，平台在顶部——机器人出生在平台上，随指令重采样在
        # 坡面上下/横向行走）。slope_range 是 rise/run：按难度 0.03→0.10 ≈
        # 1.7°→5.7°——小机器人配小坡。不用倒置版（见上面倒金字塔 env_origin 的
        # 同类穿地风险）。vertical_scale=0.001 把量化台阶控制在 1 mm，使缓坡平滑
        # 而非 5 mm 台阶的楼梯。
        # Gentle slopes (heightfield pyramid, platform on TOP — robot spawns on
        # the flat platform and walks down/up/across the slope as commands
        # resample). slope_range is rise/run: 0.03→0.10 ≈ 1.7°→5.7° by
        # difficulty — small robot, small slopes. NOT inverted (see the
        # inverted-pyramid env_origin note above — same pit-spawn risk class).
        # vertical_scale=0.001 keeps quantization steps at 1 mm so a gentle
        # slope is smooth instead of a staircase of 5 mm ledges.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,# 缓坡地形比例
            slope_range=(0.03, 0.10),# 缓坡范围（rise/run）
            platform_width=2.0,# 平台宽度（m）
            vertical_scale=0.001,# 垂直缩放因子（m）
        ),
    },
    add_lights=False,
)


def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """软化地形盒体 geom 接触，降低边缘接触导致的 NaN 不稳定。

    盒体地形中相邻 geom 高度不同；脚落在高度突变的硬边上时接触法向会
    不稳定，可能在 MuJoCo 求解器中产生冲击性 NaN 力。

    将 solref 时间常数加倍（0.02 → 0.04 s）使接触弹簧软 2 倍——足以阻尼
    该不稳定，且不会明显改变宏观行走物理。应用于 "terrain" body 的全部
    geom（TerrainGenerator 生成的所有盒子都挂在该 body 下）。

    English: Soften terrain box geom contacts to reduce edge-contact NaN instability.
    Box terrains place adjacent geoms at different heights. The hard edges where
    heights change cause contact normal instability when feet land on them, which
    can produce impulsive NaN forces in the MuJoCo solver.
    Doubling the solref time constant (0.02 → 0.04 s) makes contact springs
    2× softer — enough to damp the instability without noticeably changing the
    macro-level walking physics. Applied to all geoms in the "terrain" body,
    which contains every box generated by TerrainGenerator.
    """
    body = spec.body("terrain")# 地形体
    count = 0# 地形 geom 数量
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]   # 时间常数软 2 倍（默认 0.02）(2× softer time constant (default: 0.02))
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # 阻抗略软 (slightly softer impedance)
        count += 1# 地形 geom 数量
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")# 打印软化后的 geom 数量


def make_microduck_velocity_env_cfg(
    play: bool = False,  # 是否在训练时播放环境
    rough: bool = False,  # 是否软化地形盒体 geom 接触
) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 速度跟踪环境配置。"""

    std_standing = {
        # 下半身——更紧，站立（指令=0）时把机器人锁在 HOME 姿态
        # Lower body — tighter to keep the robot in home pose when standing
        r".*hip_yaw.*": 0.1,     # 0.1→0.06→0.05 — 保持 5° 内扣站姿（脚底放平），阻止腿外撇 (hold the 5°-inward stance (sole sits flat), stop leg splay)
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — 保持 5° 内扣站姿（脚底放平），阻止腿外撇 (hold the 5°-inward stance (sole sits flat), stop leg splay)
        r".*hip_pitch.*": 0.15, # 0.1→0.06→0.05 — 保持 5° 内扣站姿（脚底放平），阻止腿外撇 (hold the 5°-inward stance (sole sits flat), stop leg splay)
        r".*knee.*": 0.15,    # 0.1→0.06→0.05 — 保持 5° 内扣站姿（脚底放平），阻止腿外撇 (hold the 5°-inward stance (sole sits flat), stop leg splay)
        r".*ankle.*": 0.1,     # 0.1→0.06→0.05 — 保持 5° 内扣站姿（脚底放平），阻止腿外撇 (hold the 5°-inward stance (sole sits flat), stop leg splay)
    }

    std_walking = {
        # 下半身 (Lower body)
        r".*hip_yaw.*": 0.3,     # 0.1→0.06→0.05 — 保持 5° 内扣站姿，阻止腿摆到竖直 (hold the 5°-inward stance, stop the leg splay to vertical)
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — 保持 5° 内扣站姿，阻止腿摆到竖直 (hold the 5°-inward stance, stop the leg splay to vertical)
        r".*hip_pitch.*": 0.4, # 0.1→0.06→0.05 — 保持 5° 内扣站姿，阻止腿摆到竖直 (hold the 5°-inward stance, stop the leg splay to vertical)
        r".*knee.*": 0.4,     # 0.1→0.06→0.05 — 保持 5° 内扣站姿，阻止腿摆到竖直 (hold the 5°-inward stance, stop the leg splay to vertical)
        r".*ankle.*": 0.25, # was 0.15 / 曾为 0.15
    }

    site_names = ["left_foot", "right_foot"]

    # 足端接触传感器——顺序为左、右 (Contact sensor for feet - LEFT, RIGHT order)
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # 左脚在前，右脚在后 (LEFT foot first, RIGHT foot second)
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # mjlab 1.3.0：foot_height 观测与 foot_clearance/foot_swing_height 奖励
    # 现在由每只脚的地形高度射线传感器驱动（此前基于 site_pos）。
    # 对应 microban 的 foot_height_scan。
    # mjlab 1.3.0: foot_height obs + foot_clearance/foot_swing_height rewards are
    # now driven by a per-foot terrain-height ray sensor (was site_pos based).
    # Mirrors microban's foot_height_scan.
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # 基础配置 (Base configuration)
    cfg = make_velocity_env_cfg()

    # 机器人设置 (Robot setup)
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    # 动作配置 (Action configuration)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === 奖励 (REWARDS) ===
    # 姿态奖励配置 (Pose reward configuration)
    cfg.rewards["pose"].params["std_standing"] = std_standing  # 指令=0 时收紧 (tight when command=0)
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    # 姿态奖励只作用于腿部关节。头/颈由指令驱动（head_pose_tracking）——若也放进
    # 本奖励，它会把这些关节拉向 HOME，而 head_pose_tracking 拉向指令；当大指令
    # 下 head_pose_tracking 的梯度消失后，pose 奖励占优，策略收敛到"无视指令"。
    # Pose reward operates on LEG joints only. Head/neck are command-driven
    # (head_pose_tracking) — if they were in this reward too, it would pull
    # them to HOME while head_pose_tracking pulls them to the command, and the
    # policy converges to "ignore the command" because pose reward dominates
    # once head_pose_tracking's gradient dies at large commands.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # 按刚体定制的奖励配置 (Body-specific reward configurations)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # upright：刻意加强（2.0 / std²=0.05，曾为 1.0 / std²=0.1）。
    # 2026-07 俯仰-速度评估：策略以 +2-4° 恒定前倾行走（p90 约 6-8°），速度下
    # 推扰致摔约 2/3 是向前摔。权重 1.0 / std²=0.1 时 4° 前倾仅 ~0.05/步——几乎
    # 免费；2.0 / std²=0.05 时 ~0.19/步：足以在稳态步态中保持躯干水平，同时
    # 瞬态前倾（推扰恢复、加速）仍可承受。
    # upright: deliberately strong (2.0 / std²=0.05, was 1.0 / std²=0.1).
    # 2026-07 pitch-vs-speed eval: the policy walks with a +2-4° steady forward
    # lean (p90 ~6-8°) and ~2/3 of push-induced falls at speed are FORWARD. At
    # weight 1.0 / std²=0.1 a 4° lean cost ~0.05/step — effectively free. At
    # 2.0 / std²=0.05 it costs ~0.19/step: enough gradient to hold the trunk
    # level in steady gait while transient lean (push recovery, accel) stays
    # affordable.
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # 足端相关配置。mjlab 1.3.0 中 foot_swing_height 完全由传感器驱动
    # （无 asset_cfg）；只有 foot_clearance/foot_slip 仍带 asset_cfg，用
    # site_names 选脚。
    # Foot-specific configurations. In mjlab 1.3.0 foot_swing_height is fully
    # sensor-driven (no asset_cfg); only foot_clearance/foot_slip still carry an
    # asset_cfg whose site_names select the feet.
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # 刚体相关配置 (Body-specific configurations)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # foot_slip 刻意偏弱（-0.1 而非 -1.0）：-1.0 对该机器人以枢轴为主的转向
    # 限制过强。
    # foot_slip deliberately weak (-0.1, not -1.0): -1.0 was too restrictive
    # for this robot's pivot-heavy turning.
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    cfg.rewards.pop("soft_landing", None)

    # 自碰惩罚：阻止腿撞进躯干电池架（leg、leg_2、battery_holder 上
    # self_collision_only 类的 geom）。有了合理的关节限位策略其实碰不到身体，
    # 但这里的正信号能让它保持足够距离。
    # Self-collision penalty: discourages legs from crashing into the trunk
    # battery holder (the self_collision_only-classed geoms on leg, leg_2,
    # battery_holder). With proper joint-range limits the policy can't actually
    # reach the body, but a positive signal here keeps it well clear.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )


    # air_time 窗口 [0.125, 0.300] s。注意：零指令下站立不动由 standing_envs
    # 课程教学（约 2000 迭代时 →25% 站立环境），不靠显式的静止/禁止迈步项。
    # air_time window [0.125, 0.300] s. NOTE: standing still at zero command is
    # taught by the standing_envs curriculum (→25% standing envs by ~iter 2000),
    # not by an explicit stillness/no-stepping term.
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # 速度跟踪奖励 (Velocity tracking rewards)
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # 动作平滑：阶段 0 的值；下方 action_rate_weight 课程到 1500 迭代将其
    # 从 -0.1 渐增至 -1.0。
    # Action smoothness: stage-0 value; the action_rate_weight curriculum below
    # ramps it -0.1 → -1.0 by iter 1500.
    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02  # 从 0.01 上调，惩罚拖脚 (Increased from 0.01 to penalize dragging)

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02  # 从 0.01 上调，强制抬脚 (Increased from 0.01 to force foot lifting)

    # 注意：没有单独针对颈部的 action-rate 项——共享的 action_rate_l2 对所有
    # 动作维度求和（含颈部），而下面的 head_pose_tracking 已给 4 个颈/头自由度
    # 一个位置目标，颈部已被充分塑形。
    # NOTE: no neck-only action-rate term — the shared action_rate_l2 sums over
    # ALL action dims (neck included), and head_pose_tracking below gives the
    # 4 neck/head DOFs a position objective, so the neck is fully shaped.

    # 事件 (Events)
    # BAM（mjlab_frictionloss 分支）每步都会写每环境的 dof_frictionloss/
    # dof_damping；这个 no-op 事件把这些字段注册进来以便按世界展开。
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # 脚垫更抓地——从 (0.3, 1.2) 收窄 (Grippier footpad — narrowed from (0.3, 1.2))
    # 终止已数值失稳（物理 NaN）的环境。极端接触冲击可让 MuJoCo 产生 NaN 关节
    # 位置；立即终止并重置到有效状态，避免 NaN 传入观测缓冲、污染网络权重。
    # Terminate environments that have gone numerically unstable (NaN physics).
    # MuJoCo can produce NaN joint positions on extreme contact impulses.
    # Terminating immediately resets to a valid state before NaN propagates
    # into the observation buffer and corrupts network weights.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # 基于速度的推扰，用于鲁棒性训练 (Velocity-based pushes for robustness training)
    if ENABLE_VELOCITY_PUSHES:
        # play 模式用更短间隔，便于观察 (In play mode, use shorter interval for better visibility)
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # 领域随机化——每次 reset 时按 episode 重新采样。mjlab 1.3.0 中带
    # operation="add"/"scale" 的原生 dr.* 操作每次 reset 都从编译期默认字段
    # 读取（Operation.uses_defaults=True），因此天然非累积——上游这一行为
    # 取代了 microduck 旧的"先还原再加"自定义函数（当年为绕开累积坑而写）。
    # Domain randomization — re-sampled per episode at reset. In mjlab 1.3.0 the
    # stock dr.* ops with operation="add"/"scale" read from the compile-time
    # default field each reset (Operation.uses_defaults=True), so they are
    # NON-accumulating natively — this upstream behavior replaces microduck's old
    # custom restore-then-add functions that worked around the accumulation footgun.
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        # 随机化头部组件各刚体的质心（每个 reset 每刚体重新采样偏移）。
        # Randomize the CoM of the head assembly bodies (per-body fresh offset each reset).
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # 随机化电机 PD 增益 (Randomize motor PD gains)
        # 使用能处理 DelayedActuator 的自定义函数 (Uses custom function that handles DelayedActuator)
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # 物理一致的质量+惯量随机化，经由 mjlab 的 pseudo_inertia：alpha 以
        # e^(2*alpha) 同时缩放质量与惯量，质心不变（因此与 randomize_com 不
        # 冲突）。alpha_range 由 ±5% 质量缩放范围导出：e^(2*alpha) ∈ [0.95, 1.05]。
        # 取代旧的自定义 randomize_mass_and_inertia——它在 mjlab 1.3.0 下是静默
        # no-op（直接写每环境 body_mass/body_inertia 不会被展开，坍缩成单一共享
        # 值）。startup 模式 = 整个 run 内每环境固定（质量 DR 的标准做法；无累积）。
        # Physics-consistent mass + inertia randomization via mjlab's pseudo_inertia:
        # alpha scales BOTH mass and inertia by e^(2*alpha) with the CoM unchanged
        # (so it does NOT conflict with randomize_com). alpha_range is derived from
        # the ±5% mass scale range: e^(2*alpha) ∈ [0.95, 1.05].
        # Replaces the old custom randomize_mass_and_inertia, which was a silent
        # no-op under mjlab 1.3.0 (direct per-env body_mass/body_inertia writes are
        # not expanded and collapse to a single shared value). Startup mode = fixed
        # per env for the whole run (standard for mass DR; no accumulation).
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # BAM 下的关节摩擦 DR：经 FrictionDRBamActuator 的 friction_scale 钩子
        # 按环境缩放 BAM 的速度无关摩擦预算（库仑 + Stribeck + 负载项）。
        # BAM 会把 MuJoCo 的 dof_frictionloss 清零，故原生 dr.dof_frictionloss
        # 是 no-op——这才是 BAM 原生路径。
        # Joint-friction DR under BAM: scales BAM's velocity-independent friction
        # budget (Coulomb + Stribeck + load) per-env via the FrictionDRBamActuator
        # friction_scale hook. MuJoCo's dof_frictionloss is zeroed under BAM, so the
        # stock dr.dof_frictionloss is a no-op — this is the BAM-native path.
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # 随机化关节阻尼（润滑、温度效应）。自定义非累积缩放器。注意：BAM 下是
        # no-op（dof_damping 在 edit_spec 中被清零）；只影响 XML 位置执行器。
        # Randomize joint damping (lubrication, temperature effects).
        # Custom non-accumulating scaler. NOTE: no-op under BAM (dof_damping
        # zeroed in edit_spec); only affects the XML position actuator.
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # domain_randomization=True 必需 (required by domain_randomization=True)
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # 随机化转子反射惯量（armature），与 microban 完全一致
        # （dr.joint_armature，scale，±10%）。非累积（uses_defaults）。确实影响
        # BAM 执行器——BAM 设置 dof_armature（~0.0018），并未清零。
        # Randomize reflected rotor inertia (armature), microban-exact
        # (dr.joint_armature, scale, ±10%). Non-accumulating (uses_defaults). DOES
        # affect the BAM actuator — BAM sets dof_armature (~0.0018), it isn't zeroed.
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU 朝向随机化（安装误差）在下方以观测层面施加（对 projected_gravity +
    # base_ang_vel 施加每环境恒定旋转）。旧的基于事件的 randomize_imu_orientation
    # 写 site_quat，在 mjlab 1.3.0 下既不按环境展开也不被这些观测读取——no-op。
    # IMU orientation randomization (mounting error) is applied at the OBSERVATION
    # level below (per-env constant rotation of projected_gravity + base_ang_vel).
    # The old event-based randomize_imu_orientation wrote site_quat, which under
    # mjlab 1.3.0 is neither per-env expanded nor read by these obs — a no-op.

    # 躯干初始朝向随机化（迫使策略学会响应）
    # (Base orientation randomization (forces reactive behavior))
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # 观测 (Observations)
    #del Python 的字典删除语句，把这个 key 从 terms 里彻底移除  名
    # 为 actor 的观测组 —— 这是策略（Actor）网络实际能看到的观测（
    # 通常带噪声、模拟真实传感器）
    #其中名为 base_lin_vel 的观测项，即 base linear velocity，
    #机器人基座（躯干）在世界坐标系下的线速度​
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 默认给 actor/critic 两组都加 height_scan 项（地形射线扫描）。
    # microduck 没有装这种机身地形传感器，两组都删掉（对齐 microban）。
    # mjlab 1.3.0 adds a height_scan term (terrain ray scan) to both groups by
    # default. The microduck has no such body-mounted terrain sensor for the
    # policy, so drop it from both (mirrors microban).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # 仅把 base_lin_vel 加给 critic（特权信息）
    # (Add base_lin_vel to critic only (privileged information))
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,#机器人基座（躯干）在世界坐标系下的线速度​
        scale=1.0,
    )

    # 按开关确定重力/加速度计项名
    # (Determine gravity/accelerometer term name based on flag)
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # 开关为 False 时用 raw_accelerometer 替换 projected_gravity
    # (Replace projected_gravity with raw_accelerometer if flag is False)
    if not USE_PROJECTED_GRAVITY:
        # 删除 projected_gravity 并添加 raw_accelerometer
        # (Remove projected_gravity and add raw_accelerometer)
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )
    # 对 actor 组中的 projected_gravity 观测项做深拷贝，避免与其他组/配置共享引用
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # 曾为 3（最差 60 ms）；真实 dxl IMU 链路很快——±20 ms 包络（2026-07 审计）(was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit))
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1  # 曾为 3（最差 60 ms）；真实 dxl IMU 链路很快——±20 ms 包络（2026-07 审计）(was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit))
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # critic 里由传感器派生的观测项是 nan_state 保护不到的唯一路径（它只检查
    # 关节+根状态；这些项读射线/接触传感器数据，状态本身干净时 MuJoCo 也可能
    # 返回非有限值）。这里一个 NaN 就会经 rsl_rl 的 check_nan 杀死整个 run——
    # 即 2026-08-21 Velocity2-Rough-Backlash 的崩溃。只处理 critic，净化对
    # 策略零成本。
    # The critic's sensor-derived terms are the one obs path `nan_state` cannot
    # protect (it checks joint + root state; these read raycast/contact sensor
    # data, which MuJoCo can return non-finite for while the state is still
    # clean). A single NaN here kills the whole run via rsl_rl's check_nan —
    # that is the 2026-08-21 Velocity2-Rough-Backlash crash. Critic-only, so
    # sanitizing costs the policy nothing.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # 观测噪声配置（按需修改）(Observation noise configuration (edit these values as needed))
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # was 0.2 / 曾为 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15 / 曾为 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # was 0.05 / 曾为 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # was 2.0 / 曾为 2.0

    # IMU 安装失准 DR（对 IMU 派生观测施加每环境恒定旋转）。只施加给 ACTOR
    # （策略看到的是略微旋转的 IMU 系，就像真实安装误差）；critic 保留真值。
    # IMU mounting-misalignment DR (per-env constant rotation of the IMU-derived
    # observations). Applied to the ACTOR only (the policy sees a slightly rotated
    # IMU frame, like a real mounting error); the critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # joint_vel 加 1 个控制步滞后：Dynamixel 固件用上一位置采样窗口的滑动平均
    # 计算 present_velocity，所以策略实际读到的值约旧 1 个控制周期。与实物
    # 一致，并阻止策略依赖瞬时 qdot 反馈。
    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    # Matches reality and stops the policy relying on instantaneous qdot feedback.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # 从 joint_pos/vel 观测中排除 passive_* 关节（下颌连杆），使观测维与动作维
    # （14）一致而非原始铰链数（16）。先 deepcopy 各 joint_pos/joint_vel 项——
    # actor 与 critic 共享基础模板的同一项对象/params 字典，改一个会泄漏到
    # 另一个（例如下面的编码器偏置 biased 标志）。
    # Exclude passive_* joints (jaw linkage) from joint_pos/vel obs so the
    # observation dim matches the action dim (14) instead of the raw articulation (16).
    # Deepcopy each joint_pos/joint_vel term first — actor and critic share the
    # same term objects/params dicts from the base template, so mutating one would
    # leak into the other (e.g. the encoder-bias `biased` flag below).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # 编码器偏置 DR：基础模板采样每环境恒定的关节编码器偏移（startup 事件
    # "encoder_bias"），但 joint_pos_rel 在 biased=True 之前会忽略它。只把带偏
    # 的关节位置喂给 ACTOR（真实编码器读数）；critic 保留真实关节位置（特权）。
    # Encoder-bias DR: the base template samples a per-env constant joint-encoder
    # offset (startup event "encoder_bias"), but joint_pos_rel ignores it unless
    # biased=True. Feed the biased joint pos to the ACTOR only (what the real
    # encoders report); the critic keeps the true joint pos (privileged).
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # 指令——deepcopy 以免共享状态被其他 env cfg 污染
    # （make_velocity_env_cfg() 返回的对象带共享可变引用；standup/ground_pick
    # 会原地改 commands["twist"]，把范围清零）
    # Commands — deepcopy to avoid shared-state corruption from other env cfgs
    # (make_velocity_env_cfg() returns objects with shared mutable references;
    # standup/ground_pick envs mutate commands["twist"] in place, zeroing ranges)
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    command.rel_standing_envs = 0.02  # 起步即小而非零，由课程渐进上调 (small but non-zero from the start, ramped up by curriculum)
    command.rel_heading_envs = 0.0
    # 适中且固定的指令范围（不用拓宽式课程）：曾经渐进到线速度 ±0.4 / 角速度
    # ±2.0 超出了机器人能力，1000 迭代后奖励/回合长度双双下滑。角速度 ±1.0
    # 是关键变化——它让转向变得可学。
    # Modest, FIXED command ranges (no widening curriculum): a ramp to
    # lin ±0.4 / ang ±2.0 outpaced the robot's capability and tracked a
    # post-iter-1000 reward/episode-length decline. ang ±1.0 is the big
    # change — it makes turning learnable.
    command.ranges.lin_vel_x = (-0.4, 0.4)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # 显式原地旋转桶（见上方 TURN_IN_PLACE_FRACTION）。
    # (Explicit turn-in-place bucket (see TURN_IN_PLACE_FRACTION above).)
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION

    # 头部位姿指令（相对 HOME 的 4 维增量，关节顺序：neck_pitch, head_pitch,
    # head_yaw, head_roll）。作为主要奖励跟踪——见下方新增的
    # "head_pose_tracking"。初始范围小而非零，让输入神经元从第 0 步起保持
    # 活性；课程再逐步拓宽。每关节的最终上限对应该关节从 HOME 出发的机械
    # 可达增量（XML 限位减去 HOME 偏置，留 ~10% 安全裕量）：
    #   neck_pitch / head_pitch：±1.10 rad（限位 ±π/2，HOME=±20°）
    #   head_yaw               ：±1.40 rad（限位 ±π/2，HOME=0）
    #   head_roll              ：±0.31 rad（限位 ±20°）
    # Head pose command (4D deltas from HOME, in joint order:
    #   neck_pitch, head_pitch, head_yaw, head_roll). Tracked as a primary
    # reward — see "head_pose_tracking" added below. Initial ranges are small
    # non-zero so input neurons stay alive from step 0; curriculum widens them.
    # Per-joint final caps reflect each joint's mechanically reachable delta
    # from HOME (XML limits minus HOME offset, with ~10% safety margin):
    #   neck_pitch / head_pitch: ±1.10 rad (limit ±π/2 with HOME=±20°)
    #   head_yaw                : ±1.40 rad (limit ±π/2 with HOME=0)
    #   head_roll               : ±0.31 rad (limit ±20°)
    # Initial ranges are small non-zero so input neurons stay alive from step 0.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll（更紧——机械行程小得多）(tighter — much smaller mechanical range)
        ),
    )
    # 躯干位姿指令（相对标称站姿的 6 维增量：[x, y, z, roll, pitch, yaw]）。
    # 速度环境保留这个槽位以保证运行时观测形状一致；以极小权重跟踪，让输入
    # 神经元保持活性但不主导策略。standup 环境会调高权重并拓宽范围。
    # Body pose command (6D delta from nominal standing: [x, y, z, roll, pitch, yaw]).
    # Vel env carries this slot for runtime obs-shape parity; tracked at a tiny
    # weight to keep the input neurons alive but not steer the policy. The
    # standup env raises the weight + widens the ranges.
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # 把 head + body 指令观测项追加到策略和 critic 两组。顺序对运行时观测
    # 布局至关重要：[twist(3), head_pose(4), body_pose(6)]。
    # Append head + body command obs terms to both policy and critic groups.
    # Order matters for the runtime obs layout: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # === 位姿跟踪奖励 (Pose tracking rewards) ===
    # head_pose：速度环境中的主要目标——这次重写的核心。
    # std=0.5 的逐关节高斯（见 mdp.py 的 head_pose_tracking）：在满量程 ±1.0 rad
    # 指令下，完全不跟踪的策略每关节仍能拿到 exp(-(1/0.5)²)=exp(-4)≈0.018 的
    # 奖励——小而非零的梯度——课程拓宽不会杀死信号。最终奖励是 4 个关节的
    # 均值，部分跟踪得部分奖励（非全有全无）。
    # head_pose: primary objective in vel env — the whole point of the rewrite.
    # std=0.5 with per-joint Gaussian (see head_pose_tracking in mdp.py): at the
    # full ±1.0 rad command, a non-tracking policy still sees per-joint reward
    # exp(-(1/0.5)²)=exp(-4)≈0.018 — a small but non-zero gradient — so the
    # curriculum widening doesn't kill the signal. Final reward is the mean
    # over 4 joints, so partial tracking is partial reward (no all-or-nothing).
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # body_pose：基础设施保留但禁用（权重 0）——obs 槽位和指令保持活性，
    # 供调高权重的环境（standup）使用。
    # body_pose: infra kept intact but DISABLED (weight 0) — the obs slot and
    # command stay alive for envs that raise the weight (standup).
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # 头部下垂修复（2026-08-20）。行走时头以约 15° 前倾下垂（实测：run
    # ww1g2198 head_pose_tracking 1.544/2.0 → 关节平均误差 14.6°）。
    # 不要靠收紧 head_pose_tracking 的 std 来修：run 5yay13u4 试过 fine_std=0.1，
    # 策略在 300 迭代前就完全不走了（air_time 1.01 → 0.02，峰值抬脚 15 mm →
    # 2 mm，熵从 10.9 崩到 1.9）。瞬时紧容差对行走的税高达 0.77/步——占整个
    # air_time 奖励的 76%——而且不可逃脱：280 g 的头（占整机质量 38%）在迈步
    # 时必须振荡。站着不动得分更高，于是它就站着不动。
    # DC 偏置与振荡不同，是可逃脱的（把颈部指令向上偏置以抵消重力下垂），
    # 所以只对它计价：对误差的 1 s EMA 取 L1。最优解下行走策略零成本。
    # Head droop fix (2026-08-20). The head walks pitched ~15° down (measured:
    # run ww1g2198 head_pose_tracking 1.544/2.0 → 14.6° mean joint error).
    # DO NOT fix this by tightening head_pose_tracking's std: run 5yay13u4 tried
    # fine_std=0.1 and the policy stopped walking entirely by iter 300 (air_time
    # 1.01 → 0.02, peak foot height 15 mm → 2 mm, entropy collapsed 10.9 → 1.9).
    # An instantaneous tight tolerance taxes walking 0.77/step — 76% of the whole
    # air_time reward — and is UNESCAPABLE, since a 280 g head (38% of robot
    # mass) must oscillate while stepping. Standing still scored higher, so it
    # stood still.
    # The DC bias, unlike the oscillation, IS escapable (bias the neck command up
    # to cancel gravity sag), so price only that: L1 on a 1 s EMA of the error.
    # At the optimum this costs a walking policy nothing.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # 由下方 head_pose_bias_weight 课程渐进 (ramped by the head_pose_bias_weight curriculum below)
        params={"command_name": "head_pose", "tau_s": 1.0},
    )

    # 地形 (Terrain)
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG

        # 软化地形盒体接触：高度不同的相邻盒体形成硬边，会破坏接触求解器
        # 稳定性并产生 NaN 力。
        # Soften terrain box contacts: adjacent boxes at different heights create
        # hard edges that destabilise the contact solver and produce NaN forces.
        cfg.scene.spec_fn = _soften_terrain_contacts

        # 速度环境默认 nconmax=35 对粗糙地形太紧：机器人摔倒时多个身体连杆
        # 同时撞上多个盒子，接触数溢出 → 部分被静默丢弃 → 突然失压 → NaN。
        # The velocity env default nconmax=35 is tight for rough terrain: when the
        # robot falls and multiple body links hit multiple boxes simultaneously,
        # contacts overflow → some are silently dropped → sudden decompression → NaN.
        cfg.sim.nconmax = 200   # was 35 / 曾为 35

        # 速度环境只用了 10 次求解器迭代（默认 100），对粗糙盒体地形上的边缘
        # 接触太少。迭代数增至 3 倍可显著减少接触求解失败，GPU 上算力代价
        # 适中（MJWarp 按环境并行）。
        # The velocity env uses only 10 solver iterations (vs the default 100),
        # which is too few to resolve edge contacts on rough box terrain.
        # Tripling iterations significantly reduces contact resolution failures
        # with a modest compute cost on GPU (MJWarp parallelises across envs).
        cfg.sim.mujoco.iterations = 30    # was 10 / 曾为 10
        cfg.sim.mujoco.ls_iterations = 50  # was 20 / 曾为 20

        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # action_rate 权重渐进：步态 bootstrap 期间平滑温和，之后到 1500 迭代
    # 收紧到 -1.0。
    # action_rate weight ramp: gentle smoothing while the gait bootstraps, then
    # tighten to -1.0 by iter 1500.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # 步态建立后逐步提高站立环境比例
    # (Gradually increase standing env fraction after walking is established)
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0,           "rel_standing_envs": 0.02},
                {"step": 500 * 24,    "rel_standing_envs": 0.05},
                {"step": 750 * 24,    "rel_standing_envs": 0.1},
                {"step": 1000 * 24,   "rel_standing_envs": 0.15},
                {"step": 1500 * 24,   "rel_standing_envs": 0.2},
                {"step": 2000 * 24,   "rel_standing_envs": 0.25},
            ],
        },
    )

    # 注意：没有速度指令范围课程——范围固定（见上方指令部分）。
    # NOTE: no velocity-command-range curriculum — ranges are fixed (see the
    # command section above).

    # 头部位姿指令范围课程——逐关节、按各关节从 HOME 出发的可达增量设定
    # （相对 XML 限位留 ~10% 裕量）。与之前相同的 5 阶段形状（每关节最终上限
    # 的 5% → 15% → 35% → 65% → 100%）。neck/head pitch 最终 ±1.10 rad，
    # head_yaw ±1.40，head_roll ±0.31。
    # Head pose command range curriculum — per-joint, scaled to each joint's
    # reachable delta from HOME (with ~10% margin from XML limits). Same 5-stage
    # shape as before (5% → 15% → 35% → 65% → 100% of each joint's final cap).
    # neck/head pitch final ±1.10 rad, head_yaw ±1.40, head_roll ±0.31.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,                ranges = ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # 躯干位姿指令范围课程：速度环境保持小范围。standup 环境会用宽范围 +
    # 高奖励权重覆盖此课程。
    # Body pose command range curriculum: stay small in vel env. Standup env
    # overrides this curriculum with wide ranges + heavy reward weight.
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {"step": 0, "ranges": (
                    (-0.005, 0.005),  # x (m)
                    (-0.005, 0.005),  # y (m)
                    (-0.005, 0.005),  # z (m)
                    (-0.05, 0.05),    # roll
                    (-0.05, 0.05),    # pitch
                    (-0.05, 0.05),    # yaw
                )},
            ],
        },
    )

    # 质心随机化范围课程——从小开始渐进
    # (CoM randomization range curriculum - start small, ramp up)
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    # 上限 ±15 mm（2026-07 审计）：此前渐进到 ±30 mm 超出了脚的
                    # 支撑多边形（脚跟只在踝后 20 mm）——随机化的质心可能完全落
                    # 在支撑区之外，逼出宽/快的过度反应步态，并使向后平衡无法
                    # 训练。退化时间线与范围递增吻合：0.015 → 0.02 → 0.03，
                    # 策略随之越来越差。
                    # Capped at ±15 mm (2026-07 audit): the previous ramp to ±30 mm
                    # exceeded the foot support polygon (heel is only 20 mm behind
                    # the ankle) — the randomized CoM could sit entirely outside
                    # support, forcing a wide/fast hyper-reactive gait and making
                    # BACKWARD balance untrainable. Regression timeline matched the
                    # ramp increases: 0.015 → 0.02 → 0.03 as policies got worse.
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                    {"step": 1500 * 24,  "range": 0.015},
                ],
            },
        )

    # 头部质心随机化范围课程——从小开始渐进
    # (Head CoM randomization range curriculum - start small, ramp up)
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    # 上限 ±10 mm（2026-07 审计——与躯干质心同样的过度保守担忧；
                    # 头是一个大杠杆臂）。
                    # Capped at ±10 mm (2026-07 audit — same over-conservatism
                    # concern as trunk CoM; head is a large lever arm).
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                ],
            },
        )

    # 关闭默认课程 (Disable default curriculum)
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # head_pose_bias 渐进：600 迭代前关闭，之后到 1500 迭代从 1.0 → 3.0。
    # 早期保持在 0，因为步态尚未存在时姿态精度项只是干扰。权重 3.0 时 15°
    # 残余偏置每步代价 0.79，2° 偏置每步 0.10。
    # head_pose_bias ramp: OFF until iter 600, then 1.0 → 3.0 by iter 1500.
    # Held at 0 early because a posture-precision term is a distraction before
    # a gait exists. At weight 3.0 a 15° residual bias costs 0.79/step and a
    # 2° bias costs 0.10/step.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )

    return cfg


MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velocity",  # 目录名 (Directory name)
    run_name="velocity",  # 追加在 wandb 时间戳后：<datetime>_velocity (Appended to datetime in wandb)
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
