# 设计 — Roller Crouch-Glide（« 按键触发的蹲下滑行 »）

**日期：** 2026-07-17
**状态：** 设计已验证，可进入实现计划阶段

## 背景

microduck 机器人已经会滑行（roller policy，任务 `Mjlab-Velocity-Flat-MicroDuck-Rollers`）。
我们想要一个新动作：按键触发时，它**下蹲并依靠惯性继续滑行**
（像滑冰者保持低姿势），维持约 1 s，然后**自己站起来**
并恢复滑行。

使用方的硬性约束：**不修改 Rust runtime**
（`apirrone/microduck_runtime`，以二进制安装）。因此该动作必须复用
runtime 中已有的机制。

**关键发现：** runtime 已经有一个「按键触发的 one-shot 行为」slot：
`--ground-pick`。它由 **A 键**（上升沿）触发，
执行一个由 **phase** 控制的 ONNX policy，持续固定时长，然后自动
返回主 policy。尤其重要的是，它使用**与 roller policy 完全相同的
61D 观测布局** — 两者在 runtime 中可互换。这是理想的载体，无需改动
一行 Rust。

已接受的折中：该动作是 **one-shot**（固定时长，没有「保持切换」）。
下蹲的时长由 slot 的周期决定。

## 选定方案（方案 B）

创建一个**新的 mjlab 任务**，在 rollers 机器人上训练，
执行 下蹲 → 蹲着滑行 → 起身，由 ground-pick slot 的 phase 驱动。
将其导出为 ONNX 并通过 `--ground-pick` 加载。无需修改 Rust。

### 涉及的文件

| 文件 | 操作 |
|---|---|
| `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py` | **新建。** 混合 roller + ground-pick 的 env。 |
| `src/mjlab_microduck/tasks/mdp.py` | **新增** `crouch_glide_height_by_phase` reward。 |
| `src/mjlab_microduck/tasks/__init__.py` | **新增**：注册 `Mjlab-RollerCrouch-Flat-MicroDuck`。 |

### 复用（不要重新发明）

- **物理 / roller 机器人** ← `microduck_velocity_rollers_env_cfg.py`：
  `MICRODUCK_WALK_ROLLERS_ROBOT_CFG`（14 个主动关节 + 4 个被动轮），
  `roller_blade` 上的接触传感器，轴承摩擦 DR
  （`randomize_wheel_friction` + curriculum），14 维 obs（通过
  `SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))` 排除车轮），
  `action.scale=1.0`，`kp_fw=200`。
- **phase / one-shot 机制** ← `microduck_ground_pick_env_cfg.py`：
  命令 `microduck_mdp.GroundPickPhaseCommand` **原样复用**
  （生成 `[cos(2πφ), sin(2πφ), 0]`，runtime 会将其放入 twist slot），
  head/body 零填充（`zero_command_padding`），终止 `robot_state_is_nan`，
  `reset_action_history`。
- **sim2real DR** ← 从 roller env 原样复用（obs 级 IMU misalignment、
  encoder bias、质量/惯量、BAM 摩擦、armature、轻推 ±0.2）。

## 核心：由 phase 驱动的「梯形」高度目标

这是唯一真正的创新点。与 ground-pick 把嘴压到地面不同，我们根据
phase 控制**躯干高度**（`trunk_base` 的 `com_height`），并设一个低位平台：

```
hauteur
 haute ┐                    ┌──   debout (rend la main à la policy roller)
       │ \                 /
  basse│  \_______________/       accroupi + glisse (palier 1 s)
       └───────────────────────► phase
       0   0.375      0.625   1
```

- φ ∈ [0, 0.375]：下降到下蹲高度
- φ ∈ [0.375, 0.625]：**保持下蹲**（在 4 s 周期中占 1 s）→ 滑行
- φ ∈ [0.625, 1.0]：起身回到 roller 站立 pose

**在 `mdp.py` 中新增 reward `crouch_glide_height_by_phase(env, command_name, height_low,
height_high, hold_lo=0.375, hold_hi=0.625, std=...)`**：
从命令读取 phase，计算目标高度（在 高→低→高 之间插值，
在平台段保持平直），奖励 `exp(-((h_mesurée - h_cible)/std)²)`。
参考已有的 `com_height_target`（mdp.py:694）和
`interpolated/multistage height target` 函数。

起始值：`height_high ≈ 0.11` m（roller 站立高度，见
roller `com_height_target` 区间 0.0935–0.1235），`height_low ≈ 0.075` m（下蹲；
在 play 中微调）。phase 由命令的 `atan2(sin, cos)` 重建。

## Rewards

| Reward | 作用 | 来源 |
|---|---|---|
| `crouch_glide_height_by_phase` | 主目标（高→低→高） | **新增** |
| `wheel_speed`（权重降低至 ~2–3） | 保持惯性，下蹲时不刹车 | roller env（`wheel_speed_reward`） |
| `upright`（≈2）、`body_ang_vel`（−0.05）、`angular_momentum`（−0.02） | 平衡 / 稳定性 | roller env |
| `return_pose`（phase 结束时） | 收敛到 roller 站立 pose 以便干净地交还控制权 | 改编自 `ground_pick_return_pose` |
| `feet_flat`（−2） | 刀片平放 → 稳定滑行 | roller env |
| `action_rate_l2`、`neck_action_rate_l2`、`joint_torques_l2`、`self_collisions` | 平滑 / sim2real 迁移 | 两个 env |

**明确不包含：** `braking`（不想停下）、`mouth_ground_proximity`
/ `mouth_perpendicular_to_ground`（不接触地面）、`skating_air_time` /
`single_support` / `glide`（trick 期间不跨步 — 被动滑行）。

## 训练

- `MicroduckRollerCrouchRlCfg` = `MicroduckRollersRlCfg` 的副本
  （MLP 512/256/128、ELU、obs_normalization、PPO、`experiment_name="roller_crouch"`）。
- 在 `tasks/__init__.py` 中注册：
  `register_mjlab_task(task_id="Mjlab-RollerCrouch-Flat-MicroDuck", ...)`。
- 启动：
  ```bash
  uv run train Mjlab-RollerCrouch-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 8000
  ```
- episode 要以**真实的入口速度**启动（机器人是滚过来的），
  否则它没有可在下蹲期间维持的惯性。通过 reset event
  （非零初始速度）或 episode 开头的 push 来接线。

## 导出 + 部署（精确的 runtime 标志）

导出 ONNX（normalizer 由 `export.py` 烘焙进去），然后：

```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx \
  --new-dxl-imu --kp 200 --action-scale 0.8 \
  --max-linear-vel 0.6 --max-linear-vel-backward 0.5 --max-angular-vel 0.0 \
  --ground-pick roller_crouch.onnx \
  --ground-pick-period 5.0 \
  --ground-pick-kp-ratio 1.0 \
  --ground-pick-action-scale 0.8
```

**A 键** → crouch-glide，然后自动返回 roller policy。

**训练/部署一致性陷阱（对 sim2real 很重要）：**
- `--ground-pick-kp-ratio 1.0`：默认是 **0.6**（trick 期间把 kp 降到 120）。
  我们在 kp=200 下训练 → 必须强制为 **1.0** 才能对应。
- `--ground-pick-action-scale` 必须与训练时的 `action_scale` 一致（上例为 0.8）。
- `--ground-pick-period 5.0` 必须与训练的周期/动作长度一致
  （默认 4.0，我们保留）。

## 风险与验证

- **one-shot、固定时长：** 下蹲持续 `ground-pick-period` 然后自动起身。
  没有自由保持 — 这是方案 B 已接受的限制。
- **trick 期间的惯性：** phase 取代了速度命令 → 下蹲期间**没有主动推送**。
  如果入口惯性太弱，它会减速。因此需要用真实入口速度训练。
- **验证：**
  1. 在 sim（`play`）中：它下蹲，平台段轮子继续转动，
     起身时不摔倒，并且最终 pose 干净地回到 roller 站立 pose。
  2. 在真实机器人上：以低速启动，按 A，观察。
  3. 确认 roller policy 在返回后能干净地接管。

## 开放问题 / 实现期间需确认

- `height_low`（下蹲）的确切值 — 在 play 中调节。
- 注入入口速度的最佳方式（reset event vs 初始 push）。
- `wheel_speed` 相对 `crouch_glide_height_by_phase` 的权重（既要保持惯性，
  又不能阻止下蹲）。
