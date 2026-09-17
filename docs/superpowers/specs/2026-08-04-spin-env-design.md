# Spec — 「Spin」Env（穿轮滑鞋原地快速旋转）

日期：2026-08-04。分支：`new_pre_alpha_rollers`。

> **修订（首次运行后）**：首次校准 run（500 it.）
> 显示 robot 系统性地在约 1.16 s 倒下，远早于制动段。
> 作为应对，目标减半 — `SPIN_RATE_MAX`
> 6.0 → **3.0 rad/s**，即**每周期 1 圈而非 2 圈** — 并
> 将 `spin_stay_in_place` 加固至 **−3.0**，**不加**速度 curriculum。
> 见「初始验证结果」一节了解证据及当前生效配置。

## 目标

一个新的 RL 任务，教穿轮滑鞋的 microduck 做 **spin**：
以 ~6 rad/s（360°/s）原地逆时针转 ~2 圈*（初始目标；已下调至
3 rad/s，见修订）*，然后干净地站立停下。
**由相位驱动的周期性**动作，部署在 runtime 的**一次性按钮槽**中，
如现有的 `roller_crouch` 任务。

## 框定的决策

| 问题 | 决策 |
|---|---|
| 支撑 | 轮滑鞋（`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`，4 个被动轮） |
| 驱动 | 一次性按钮槽，命令 = 相位 `[cos(2πφ), sin(2πφ), 0]` |
| 目标 | ~6 rad/s，2 圈，然后制动至停止（初始目标；已下调至 3 rad/s，见修订） |
| 入场状态 | 静止**或**低速滑行（0 → 0.3 m/s） |
| 方向 | 仅向左（正偏航，逆时针） |
| 方法 | 「结果」目标（跟踪 ω_z）+ 递减的反对称 shaping |

**runtime 约束**：槽只发 `[cos, sin, 0]` — 没有自由通道
控制旋转方向。因此 policy **总是向左转**。镜像 policy
之后可放到另一个槽（B 按钮、`--fold-policy`）。

## 目标物理机理

在 4 个被动轮上，「干净」的原地旋转靠**差速滚动**实现：
左滑刃向后，右滑刃向前（轮子**滚**而非滑）。这是一种
*反对称 swizzle*：两腿做彼此相反的动作，而非经典 swizzle 的镜像。

逆时针旋转的符号验证（参考系：x 前、y 左、z 上；ω_z > 0）：左侧一点
（+y）速度 `ω ẑ × y ŷ = −ω y x̂`，即**向后**。4 个轮子
正向前进（由 `test_wheel_direction.py` 验证），故逆时针 spin：
`ω_左轮 < 0`、`ω_右轮 > 0`，即 **`ω_D − ω_G > 0`**。

## 选定方案（C）及理由

考虑了三种方案：

- **A — 纯「结果」目标**：reward 偏航速度，让 PPO 自己找动作。本仓库
  记录的风险：懒惰最优 / 跳跃式滑行而非干净滚动。
- **B — 由姿势驱动的「指令式」目标**：两个剪刀姿势由相位插值，
  如 `roller_crouch`。若姿势选得好则*很快*学会；但 crouch 的姿势是
  **从真实机器人读取的**，而此处动作未知。需手工拼：昂贵且风险高（
  无可用扭矩的姿势什么都产生不了）。
- **C — A + 递减的反对称 shaping** ← **选定**。A 的结构，加
  两个弱 *shaping* 项注入唯一确定的物理知识（差速滚动），
  其权重由 curriculum 递减，让 policy 精炼自己的动作。**泵频保持自由**。

## 架构

**文件**：`src/mjlab_microduck/tasks/microduck_spin_env_cfg.py`
- 工厂 `make_microduck_spin_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`
- PPO 配置 `MicroduckSpinRlCfg`
- task id `Mjlab-Spin-Flat-MicroDuck`，注册于 `tasks/__init__.py`

克隆 `microduck_roller_crouch_env_cfg.py` 的结构：rollers robot、统一 61D obs、
完整 DR、`action.scale = 1.0`、平地。

**`ENABLE_SYMMETRY = False`** — 必须：左右对称增广会把左旋变右旋并摧毁学习。

**命令**：`GroundPickPhaseCommandCfg(period=4.0, randomize_phase=False)`。
`period=4.0` 是 `--ground-pick-period` 的默认 → runtime 无需传参。
`randomize_phase=False` → 每集从 φ=0 起步（站立），如部署时。

## 相位包络

相位驱动一个**目标偏航速度** ω\*(φ)，梯形分 4 段
（周期 4 s，`SPIN_RATE_MAX = 6.0` rad/s — 初始目标；已下调至 3 rad/s，
见修订；段与周期未变）：

```
ACCEL_END = 0.125   [0,     0.125)  0.5 s  ω* : 0 → 6 rad/s   (lancement, rampe linéaire)
HOLD_END  = 0.525   [0.125, 0.525)  1.6 s  ω* = 6 rad/s        (régime)
BRAKE_END = 0.650   [0.525, 0.650)  0.5 s  ω* : 6 → 0          (freinage, rampe linéaire)
            1.0     [0.650, 1.0)    1.4 s  ω* = 0              (repos debout)
```

*(以上 ω\* = 6 rad/s 对应 `SPIN_RATE_MAX` = 6.0，初始目标；
见修订了解生效值。)*

一周期积分：`0.5·3 + 1.6·6 + 0.5·3 = 12.6 rad ≈ 2.0 圈`。✅
*（在 `SPIN_RATE_MAX = 6.0` 时，初始目标。）* 一般形式：积分值
`2.1 × SPIN_RATE_MAX`，与 `rate_max` 无关（0.25 + 1.6 + 0.25 = 2.1）。
在生效目标（3.0 rad/s）下：`2.1 × 3.0 = 6.3 rad ≈ 1 圈`/周期 — 见修订。

一集 = 20 s = **5 个周期**：robot 重复 启动 → 稳态 → 制动 → 静止
五次。每集更多数据，且「静止」段也训练 trick 的干净退出。**注（run 后）**：
几何上仍成立（20 s / 4 s），但校准 run 中无一集活过 ~1.16 s，
即仅第一周期的一小段 — 见「初始验证结果」。

**纯函数** `spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)`
在 `mdp.py` 中，紧邻 `crouch_pose_blend`。无模拟器即可测试。

**shaping 门**：`gate(φ) = spin_rate_by_phase(φ) / rate_max ∈ [0, 1]`。静止段
为 0 → 该时段无 shaping 推动剪刀，故 robot 回到中性站立。这正是
向 roller policy 干净退出 trick 的来源。

## Reward

### mjlab 中已验证的陷阱（需显式处理）

- `body_ang_vel`（`body_angular_velocity_penalty`）只惩罚 **x/y**
  （`ang_vel_xy`，注释 « Don't penalize z-angular velocity »）→ **保留**
  （权重 −0.05）：压制滚转/俯仰摆动而不干扰 spin。
- `angular_momentum`（`angular_momentum_penalty`）惩罚角动量的 **3D 范数**
  → 会直接对抗 spin。**删除。**

### 新 reward（需写入 `mdp.py`）

| Reward | 权重 | 定义 |
|---|---|---|
| `spin_rate_track` | 6.0 | `exp(−((ω_z − ω*(φ))/std)²)`，`std = 1.5` rad/s。ω_z = 躯干在体系下的偏航（IMU 所见）。主目标。 |
| `spin_rate_l1` | 0.5 | `−\|ω_z − ω*(φ)\|`：远离目标高斯饱和时的常量梯度 bootstrap（与 `crouch_glide_pose_l1` 同技巧） |
| `spin_stay_in_place` | −3.0（初始 −1.0，见修订） | 躯干 `‖v_xy‖²` → 「原地」，并杀掉入场动量。无参考状态，故对每集 5 周期稳健 |
| `spin_wheel_differential` | 1.0 | `gate(φ) · tanh(clamp(ω_D − ω_G, min=0) / omega_scale)`，其中 `ω_G = (LF+LR)/2`、`ω_D = (RF+RR)/2`：奖励与逆时针一致的反向滚动的滑刃 → **以滚动**而非滑动旋转。轮子按名解析（`passive_LF_?wheel` 等）。生效 `omega_scale = 17.0` rad/s（见下方标定段） |
| `leg_antisymmetry` | 1.0 → 0.25 | `gate(φ) · (−mean\|q_G − q_D\|)` 作用于 `hip_pitch` 和 `knee`。⚠️ 镜像约定：*对称*姿势 `q_G + q_D ≈ 0`，故**剪刀**是 `q_G ≈ q_D`。由 curriculum 递减 |
| `spin_grounded` | 0.5 | `gate(φ) · 1[n_contact ≥ 2]`：两刀接地，阻止「跳起空中扭」。swizzle 的 `grounded_reward` 不能直接复用（它按 `cmd_x` 加权，而此处 `cmd_x = cos(2πφ)`） |

**`omega_scale` 标定**（tanh 饱和尺度）：在目标稳态下，
每滑刃速度 `v = ω_z · 半轮距`，故每轮转速 `v / r`，`r = 0.0175` m，
差速为 `2 · ω_z · 半轮距 / r`。
rollers 模型中腿根在 `y = ±0.0175` m，但滑刃更分开（踝偏置）：真实半轮距
需在**首次 run 时从 `left_foot` / `right_foot` site 实测**。半轮距估计 ~0.03 m、
`ω_z = 6` rad/s 时，预期差速 ~20 rad/s — 故初始默认 `omega_scale = 20.0`。
**实测（Task 3）：真实半轮距 = 0.0499 m，预期差速 = 34.2 rad/s，比估计高 71 %
— 超出计划设定的 30 % 阈值。** 故 `SPIN_WHEEL_OMEGA_SCALE` 校正为 **34.0**
（中间值，目标 6 rad/s 时生效；后重标为 **17.0**，见下方「更新」段）。
半轮距实测细节见下方「初始验证结果」。

**更新（评审后 wave 修复）**：`SPIN_RATE_MAX` 从 6.0 降至
**3.0 rad/s**（人为决定，无 curriculum — 见下）。对 `omega_scale` 的直接
力学后果，不是独立选择：稳态预期差速重回
`2 · 3.0 · 0.0499 / 0.0175` = **17.1 rad/s**。保留
`omega_scale = 34.0` 会把该项压在 `tanh(17.1/34) = 0.47` 自身上限，恰好
削弱我们想加强的 shaping。故 `SPIN_WHEEL_OMEGA_SCALE` 重校为 **17.0**，
半轮距实测（0.0499 m）作为参考保留。

### 从 `roller_crouch` 复用的 reward（稳定性 / sim2real）

| Reward | 权重 |
|---|---|
| `upright`（躯干垂直） | 2.0 |
| `feet_flat`（刀平贴） | −2.0 |
| `self_collisions` | −1.0 |
| `body_ang_vel`（仅 xy） | −0.05 |
| `action_rate_l2` | −1.0（curriculum −0.5 → −1.0） |
| `neck_action_rate_l2` | −0.5 |
| `joint_torques_l2` | −1e-3 |
| `neck_joint_pos_l2` **不含 `head_yaw`** | −0.2 |

**头部**：颈俯仰/滚转保持近中性（sim2real），但 `head_yaw`
**排除**出该项 → 自由用作惯性飞轮启动旋转。实现：`neck_joint_pos_l2`
硬编码用正则 `.*(neck|head).*` 解析关节；故要么给该函数加正则参数，
要么写变体 `neck_joint_pos_l2_no_yaw`。选择：**给 `neck_joint_pos_l2`
加 `pattern` 参数**（默认不变）以免重复。

## Reset / 入场状态

```python
cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)
cfg.events["reset_base"].params["velocity_range"] = {"x": (0.0, 0.3)}
```

通过 `reset_root_state_uniform` 注入。**绝不**在 `mode="reset"` 用
`push_by_setting_velocity`：那是 crouch NaN 的根源（`root_vel +=` 叠加在
可能发散的根速度上 → 基座 free-joint 爆炸）。

## 域随机化

与 `roller_crouch` 相同，不偏离（仓库已验证的 sim2real 配方）：躯干
+ 头部 COM、质量/惯性、BAM 关节摩擦、armature、轮摩擦、
0.2 m/s 推力每 3–6 s、IMU 失准 6°、编码器偏置 ±0.015 rad。

## 观测

**61D 布局** 与 roller / ground_pick / crouch 一致 — ONNX 在槽中加载的
条件：
`[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`
其中 `command = [twist(3), head_pose(4), body_pose(6)]`，head/body zero-padded。

故：从 actor 移除 `base_lin_vel`（critic 保留），移除 `height_scan` 和
`foot_height`，critic 侧 `wheel_vel`，被动关节排除出 `joint_pos`/`joint_vel`，
延迟与噪声与 crouch 相同。

gyro 在 obs 中 → policy **观测**自己的 ω_z：任务可观测。

## 终止

`time_out`、`fell_over`、`out_of_terrain_bounds`（继承）+ `nan_state`
（`microduck_mdp.robot_state_is_nan`），如 crouch。

## Curriculum

| 项 | 阶段 |
|---|---|
| `action_rate_weight` | −0.5 (0) → −0.8 (250 it.) → −1.0 (500 it.) |
| `leg_antisym_weight` | 1.0 (0) → 0.5 (1500 it.) → 0.25 (3000 it.) |
| `com_range` | 0.003 → 0.005 (500 it.) → 0.01 (1000 it.) |
| `head_com_range` | 0.003 → 0.005 (500 it.) → 0.01 (1000 it.) |

（iterations × 24 steps/env，与其他 env 一致）

**目标速度无 curriculum**：一开始就 6 rad/s*（初始目标；
下调至 3 rad/s，仍无 curriculum，见修订）*。见「Plan B」。

## PPO

`MicroduckSpinRlCfg` = `MicroduckRollerCrouchRlCfg` 的副本：actor/critic
`(512, 256, 128)` elu、obs normalization、PPO adaptive lr 1e-3、`desired_kl=0.01`、
`num_steps_per_env=24`、`symmetry_cfg=None`、`experiment_name="spin"`、
`run_name="spin"`、`max_iterations=8000`。

## 测试

`tests/test_spin.py` — 纯函数，无模拟器：
- `spin_rate_by_phase`：4 段边界值（0, rate_max, rate_max, 0, 0）
- 启动 ramp 单调递增、制动单调递减
- **一周期积分 ≈ 4π** 在 `rate_max = 6.0`（保证梯形**形状**，
  `2.1 × rate_max` rad/周期）— 修订后不再保护生效目标，见下条。
  包络精确值：12.6 rad 对 4π = 12.566 → 容差 1 %
- **实际派发的目标**（`mdp.SPIN_RATE_MAX`）积分确实为
  `2.1 × SPIN_RATE_MAX` rad/周期，与 `rate_max` 无关 — 在
  7d916aa 中加入，正是此测试在目标改变而圈数未加思考时失败。生效值
  （3.0 rad/s）下：6.3 rad ≈ 1 圈
- `gate(φ) = 0` 遍历整个静止段，处处 `∈ [0,1]`

`tests/test_spin_cfg.py` — env 可构建：
- 命令 = `GroundPickPhaseCommand`、`period == 4.0`、`randomize_phase is False`
- `"angular_momentum" not in cfg.rewards`（rewards 节的陷阱）
- `symmetry_cfg is None`
- actor obs 维度 == 61
- **observation 项顺序的精确对等**（actor + critic）与
  `roller_crouch` 逐组一致 — 在 7d916aa 中加入，是导出的 ONNX 能在
  runtime 槽加载的严格条件

运行：`uv run --with pytest pytest tests/ -q`

## 训练 / 部署

```bash
uv run train Mjlab-Spin-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 8000
# surveiller Episode_Reward/spin_rate_track (doit monter)
uv run scripts/play_latest.py     # alias md-play
uv run scripts/export_latest.py   # ONNX, normaliseur d'obs baké
```

```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx --new-dxl-imu --kp 200 --action-scale 0.8 \
  --ground-pick spin.onnx \
  --ground-pick-period 4.0 \      # = SPIN_PERIOD
  --ground-pick-kp-ratio 1.0 \    # défaut 0.6 -> forcer 1.0 (entraîné kp 200)
  --ground-pick-action-scale 0.8  # matcher action_scale runtime
```

**A** 按钮 → spin，然后自动回到 roller policy。

## 成功准则

play 中：约 2.6 s 内逆时针 ~2 圈、躯干漂移 < ~10 cm、全程站立、
下一周期前静止段中性站立稳定。*（为初始目标 6 rad/s / 2 圈制定；
3 rad/s 下见修订，约稳态期间 1 圈 — 准则未修订，robot 尚未撑到那时。）*

## 若训练卡壳的 Plan B

按序：
1. **速度 curriculum**：`SPIN_RATE_MAX` 3 → 6 rad/s（需让 `rate_max` 可由
   reward params 的 `CurriculumTermCfg` 控制）。
   **部分跟进**：校准 run 后目标已下调至 3 rad/s（见修订），但
   **无 curriculum** — 3 rad/s 暂为固定目标，而非向 6 渐升的起点。
   人选择先看 robot 在该速度下能做到什么，再考虑渐升。
2. 提高 `spin_wheel_differential` 并推迟 `leg_antisymmetry` 递减。
3. 放宽 `spin_rate_track` 的 `std`（1.5 → 2.5）以在更远处获得有用梯度。
4. 最后手段，切到方案 B（在 pose editor 中手工拼剪刀姿势）以
   引导动作，再释放。

## 范围外

- 右旋（镜像 policy 放另一槽）— 之后。
- 脚式变体（无 rollers）。
- 连续速度命令式 spin（需 runtime 命令通道）。

## 初始验证结果

### 实测半轮距与 `omega_scale`

半轮距在 rollers 模型的 `left_foot` / `right_foot` site 上实测：
**0.0499 m**，对比 spec 估计的 0.03 m。稳态（6 rad/s）预期轮差速：
`2 · 6.0 · 0.0499 / 0.0175` = **34.2 rad/s**，
比默认 20.0 高 71 % — 超出计划设定的 30 % 阈值。
故 `SPIN_WHEEL_OMEGA_SCALE` 从 20.0 改为 **34.0**。测试仍显式传
`omega_scale=20.0`，以独立于常量。

### Smoke run（Step 2：5 it.、64 envs、NaN 守卫）

无异常完成。`Episode_Termination/nan_state` 全程保持 0.0000，
`/tmp/mjlab/nan_dumps/` 从未创建。6 个 spin reward 都出现在
logged 的 `Episode_Reward/` 键中：`spin_rate_track`、
`spin_rate_l1`、`spin_stay_in_place`、`spin_wheel_differential`、`spin_grounded`、
`leg_antisymmetry`。

观测对等（Step 1）：spin env 的 actor obs 项列表与 `roller_crouch`
**完全一致** — 8 项、同序：
`base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command,
head_command, body_command`。这是导出的 ONNX 能在 runtime 槽加载的条件。

**保留的使用提示**：计划中的示例命令以裸 flag `--enable-nan-guard`
会被本仓库 CLI 拒绝 — 必须传 `--enable-nan-guard True`。

### 校准 run 500 it.（Step 3）

4096 envs、500 iterations、~2.32 s/iter、退出码 0、wandb logger（故
`scripts/play_latest.py` / `md-play` 能找到该 run）。

**真正确立的事实**：`Mean episode length` = **57.83 步**，
每集 1000 步（50 Hz 下 20 s），即 **~1.16 s**。
`Episode_Termination/fell_over` ≈ **70**、`time_out = 0.0000`、
`nan_state = 0`。robot **每集都倒**，相位 φ ≈ 0.29 — 正在稳态段中段。
从未抵达制动（φ ≥ 0.525）或静止（φ ≥ 0.650）：**71 % 的周期从未被训练**。

集长从 23.98 升至 57.83 步：`Episode_Reward/spin_rate_track`
（0.0291 → 0.3168）的上升主要反映**存活时长变长**，而非跟踪变好。
该步在计划中所述的成功准则（「曲线应上升」）对该项**不是有效信号**：
完全静止的 robot 已能得 `6.0 × 0.405 = 2.43` — 静止段为「站着不动」
付全额，故任何存活更久的 policy 机械地多赚该段，与跟踪质量无关。

### 衍生诊断 — 估计值，非直接测量

以下值来自最后一段日志中 reward 项间比值，抵消了 logger 施加的未知归一化。
应视为估计，可由同一方法复现：

**撑得住的部分**：站立的 ~1.2 s 内 robot 相当贴近目标。比值
`spin_rate_l1 / spin_rate_track`（−0.0097 / 0.3168，权重 0.5 与 6.0，
`std = 1.5`），解 `e = 0.3674 · exp(−(e/1.5)²)`：偏航速度跟踪平均绝对
误差 ≈ **0.35 rad/s**，由两条独立路径确认 — 该
`spin_rate_l1 / spin_rate_track` 比值，及从 reward manager 归一化反算。
它**能启动** spin；**无法在 spin 时站立**。

**撑不住的部分**：shaping 块（`spin_wheel_differential` 1.0、
`spin_grounded` 0.5、`spin_stay_in_place` −1.0）合计 ~1.0 权重对主目标 6.0 —
约为滑行 policy 通过忽略该块所能放弃的 **13 %**。且
`spin_wheel_differential` 对瞬时旋转中心**不变**：中心化 spin 在 6 rad/s
与左滑刃支点 pivot 在 6 rad/s 都产生差速 34.2 — 故该项**未编码**
中心化滚动，仅 `spin_stay_in_place` 做。`spin_stay_in_place` ≈ −0.0069 意味
`‖v_xy‖ ≈ 0.35 m/s`：robot 仍在平移，与偏心支点（滑刃作 pivot）
而非绕身体中心旋转一致。

### 基于此诊断决定的配置变更

目标减半 — `SPIN_RATE_MAX` 6.0 → **3.0 rad/s** — 并
`spin_stay_in_place` 加固 −1.0 → **−3.0**（见 reward 表及
`SPIN_WHEEL_OMEGA_SCALE` 重标 17.0 于上）。**刻意对目标速度无
curriculum**：这是半速下 robot 能做到什么的首次尝试，再视情况考虑
渐升。

**启动段漂移代价的削弱。** 将 `spin_stay_in_place` 加固至 −3.0 让评审
指出的缺陷更尖锐：该项是 spin 中唯一不被相位调制的，故在启动 ramp
期间对瞬态平移全额计费 — 而那正是 robot 需蹬地注入角动量的时刻，
入场动量（高达 0.3 m/s）须**转换**为旋转。该代价现乘以
`SPIN_LAUNCH_DRIFT_SCALE = 0.2`（仅 `[0, ACCEL_END)`），之后全额。
**有意**不在静止段熄火，与 shaping 不同：那里不动才是真准则。

Step 4（看动作）待做，留给人。

⚠️ 这四个测试（三个关于削弱的新增、一个修改）**未**
执行 — 提交时机器另作他用。任何长 run 前须运行：
`uv run --with pytest pytest tests/test_spin.py tests/test_spin_cfg.py -q`。
