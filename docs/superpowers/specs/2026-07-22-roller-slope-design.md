# 坡道模式 — `roller_slope`（被动平衡下坡）

日期：2026-07-22
状态：设计已验证，可进入实现计划阶段。

## 目标

训练一个专用 policy，让 **microduck（穿 rollers）从平地起步，
带一个向前的微小冲量，滚到一个下坡斜面前，然后顺势滑到坡底
并保持站立和平衡**。下坡过程中不做任何驱动：policy 的唯一目标
就是**不摔倒**。

policy 需要通过 difficulty curriculum 处理坡度递增的斜面
（**约 2° → 约 20°**）。

## 已框定的决策（头脑风暴）

| 主题 | 决策 |
|---|---|
| 行为 | 被动平衡下坡（重力推动前进，不强制蹬踏） |
| 驱动 | 无 — 纯平衡，`twist` 命令强制为零 |
| 方案 | **A** — 独立的专用任务（类似 `roller_crouch`） |
| 地形形状 | **简单斜面**：起步平地 + 下坡斜面（不是金字塔） |
| episode 场景 | 在平地 spawn → 向前冲量速度 → 在斜面上滑行 |
| 坡度 | curriculum **0/2° → 20°** |
| 部署 | flag `--slope <onnx>` + `infer_policy.py` 中的 **`Y`** 键（Y 当前空闲） |

## 架构

### 1. 新任务

文件：`src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py`，从
`microduck_velocity_rollers_env_cfg.py` 克隆。

- 同样的 roller 机器人（`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`），同样的物理、
  同样的 domain randomization / 噪声 / 延迟。
- **同样的 61D 观测**（twist + head/body 零填充）→ policy 通过
  runtime 的 `--new-cmd-obs` 路径加载，并与其他 roller policy 保持互换。
- 在 `src/mjlab_microduck/tasks/__init__.py` 中通过
  `register_mjlab_task` 注册，带一个 PPO config `MicroduckRollerSlopeRlCfg`
  （`experiment_name`/`run_name` = `roller_slope`）。

### 2. 「平地 + 斜面」地形（自定义）

mjlab 自带的倾斜地形是金字塔；因此我们写一个专用的
`SubTerrainCfg`（例如 `FlatRampTerrainCfg`），其
`function(difficulty, spec, rng)` 构造：

- 一块**起步平地**（长度约 1–2 m），机器人在此 spawn；
- 紧接着一个**下坡斜面**，其角度由 `difficulty` 在
  `[~2°, ~20°]` 上**插值**。

地形通过 `TerrainEntityCfg(terrain_type="generator", ...)` 挂载，
配一个 `TerrainGeneratorCfg` 生成多个 difficulty 等级（即多个斜面角度）。
每个环境的原点必须落在**平地区域**上，斜面在其前方。

> 实现风险，需在计划中处理：spawn 原点在平地上的定位（不在
> tile 中心），以及斜面朝向，使「前方」=「朝下」。

### 3. 命令 = 无

`twist` slot 置零：`rel_standing_envs = 1.0`，速度范围设为 0，
`rel_heading_envs = 0.0`。Head/body 保持零填充。policy 不
接收任何移动指令。

### 4. Reset & 冲量速度

- `reset_base`：在平地上静止 spawn，rollers 标称高度
  `z`（~`0.1335–0.1435`，同 roller env）。
- **入口速度**通过 `reset_root_state_uniform` 的 `velocity_range`
  （干净的状态 + range）注入，**不**通过
  `push_by_setting_velocity`（后者会叠加到当前状态上，可能让
  free-joint 发散 → NaN — 这是 `roller_crouch` 上已学到的教训）：
  `x ≈ (0.2, 0.5) m/s` 向前。
- episode 期间保留轻量随机 push（保持鲁棒性），同
  roller env。

### 5. Rewards

核心是「保持直立 + 自然姿态」，并防止懒惰最优解（避免它
瘫到地上以最大化稳定性）：

- `upright`（躯干垂直）— **主项**
- `alive`（每步存活 bonus）
- **标称站立 pose**：朝 HOME pose 奖励（pose 插值机制
  沿用 `roller_crouch`，但目标固定 = 站立），
  以保持正常的 rollers 站姿，而非防御性下蹲
- `feet_flat`（rollers 平放于地面）
- `body_ang_vel`、`angular_momentum`（不抖动 / 不拧转）
- `action_rate_l2`、`neck_action_rate_l2`、`joint_torques_l2`、
  `self_collisions`（平滑 + sim2real）

> 不设速度/刹车 reward：下坡是被动过程。我们不奖励
> 「跑得快」，只奖励「下坡时保持直立」。

### 6. 终止条件

- **摔倒**：`bad_orientation`（躯干过度倾斜）。
- **到达坡底**：`out_of_terrain_bounds`（机器人到达坡底
  → reset）。
- `nan_state`、超时。

### 7. difficulty curriculum（坡度）

进度从**缓 → 陡**：从近水平的斜面开始，随着成功
逐步把角度提升到 20°。

> 实现风险：标准 curriculum `terrain_levels_vel` 按
> 相对于命令速度的行进距离来晋升。这里命令为
> 零，因此**需要自定义晋升判据**：如果机器人存活 /
> 到达坡底未摔倒则晋升，如果早早摔倒则降级。

### 8. 部署 — `Y` 键

在 `scripts/infer_policy.py` 中：

- 新增 flag `--slope <onnx>`，将坡道 policy 作为额外的 session
  加载（与 `--walking` / `--standing` / `--ground-pick` 同样模式）；
- `GLFW_KEY_Y = 89`（当前**空闲** — 头部控制在 `H`）**切换**
  活动 session 到/自坡道 policy；
- 新增键盘帮助行。

不会破坏任何现有控制（不像共用 `H` 键的头部控制
那样冲突）。

## 不在范围内（YAGNI）

- 下坡时不做左/右驱动，也不刹车。
- 不上坡，不横穿斜面。
- 不做金字塔，也不做多方向地形。
- 不从现有 roller 权重 fine-tune（从头训练
  from scratch）。

## 交付物

1. `microduck_roller_slope_env_cfg.py`（env + `FlatRampTerrainCfg` + PPO cfg）。
2. 在 `tasks/__init__.py` 中注册任务。
3. `tasks/mdp.py` 中所需的自定义 reward/curriculum（站立 pose、
   等级晋升）。
4. 在 `scripts/infer_policy.py` 中接入 `--slope` + `Y` 键。
5. 纯函数的单元测试（坡度角随 difficulty 的关系、
   可能的晋升判据）。
