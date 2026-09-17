# Spec — 通过 pose 跟踪实现 RL « shoot 一脚踢球 » 任务

**日期** : 2026-07-24
**分支** : `new_pre_alpha_ground_pick`
**Task id** : `Mjlab-Shoot-Flat-MicroDuck`

## 目标

通过**跟踪 4 个 keyframe 的关节 pose 轨迹**（按相位插值），学习一个 **shoot 一次性**（踢球）动作：

```
STAND → PIED_ARRIÈRE (armement) → PIED_AVANT (frappe) → STAND (repos)
```

- **右腿**踢球，**左腿**支撑。
- **不模拟真实球**：通过 pose 跟踪学习 *动作*（类似于 `ground_pick` / crouch）。如果在部署时机器人前方真有一颗球，球会被踢到。
- Obs **61D 统一**，与其它 microduck policy 相同 → 导出的 ONNX 可原样部署到 runtime 的**按钮 slot**（一次性：播放该动作后交还主 policy）。

与该分支的 `ground_pick` 任务使用同一模板（相位以 `[cos, sin, 0]` 编码进 twist slot，按相位跟踪 pose，obs 61D，继承自 velocity 的 sim2real DR）。

## 非目标 (YAGNI)

- 没有物理球，没有 contact/球速 reward。
- 不支持可配置方向（仅右脚；左脚如有需要可后续对称化）。
- 不行走/不恢复摔倒：移除所有 locomotion 项。

## 架构

### 文件与注册
- `src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py`
  - `make_microduck_shoot_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`
  - `MicroduckShootRslCfg` (RslRlOnPolicyRunnerCfg, `experiment_name="shoot"`)
- 在 `src/mjlab_microduck/tasks/__init__.py` 中注册：
  `Mjlab-Shoot-Flat-MicroDuck`（可选 `-Rough-` 变体）。
- 基类：继承 velocity env（通过 `make_velocity_env_cfg`，与 ground_pick 一样），然后激进剥离所有 locomotion 相关内容。
- 机器人：`MICRODUCK_WALK_ROBOT_CFG`（标准行走模型，14 关节，无 rollers）。
- `action.scale = 1.0`。

### Poses（占位 → 通过 `read_pose.py` 从真机读取）
字典 `{关节名: rad}`，**14 关节**（不含 mouth）。位于 env 文件顶部。
- `STAND_POSE`：中立站姿（约 sim 的 HOME）。
- `KICK_BACK_POSE`：右髋**后伸** + 右膝屈曲（蓄力）；左腿与颈部 ≈ HOME。
- `KICK_FWD_POSE`：右髋**前屈** + 右膝伸展（踢球）；左腿与颈部 ≈ HOME。

起始时使用合理占位值（可调整），后续用真实读取值替换。

### 指令与相位
- 复用 `GroundPickPhaseCommand`：`command = [cos(2π·φ), sin(2π·φ), 0]` 写入 twist slot。
- **周期**：`SHOOT_PERIOD ≈ 2.5 s`（通过 `cfg.period` 配置）。
- **新增 `randomize_phase` flag** 在 `GroundPickPhaseCommandCfg` / `GroundPickPhaseCommand` 上：
  - 默认 `True`（向后兼容：`ground_pick` 保持当前行为）。
  - Shoot 将其设为 `False` → `reset()` 将 φ 重置为 0 而非 `rand()`。
  - 原因：每个 episode 从 STAND 起始（机器人状态 = `default_joint_pos`），φ=0 = STAND 目标 → reset 时状态/目标一致（否则 policy 被要求从静止站姿瞬间变成「踢球」pose）。
  - **一致性不变量**：`STAND_POSE` 必须等于 sim 的 reset 关节 pose（`HOME_FRAME` / `default_joint_pos`，非零：hip_pitch ±0.4579, ankle ±0.4530, hip_roll ±0.0873, neck/head_pitch 0.3491）。由 `test_stand_pose_matches_home_standing_pose` 校验。初始零占位值违反了该不变量（在最终评审后修正）。

### Reset（站立高度，无初速）
- `reset_base.pose_range.z = (0.12, 0.13)` — **绝对站立高度**（`InitialStateCfg` 的根 `pos` 默认为 (0,0,0)，故 reset 的 z = 0.12–0.13 m，不是加性偏移；值与行走的 velocity env 相同）。无坠落。
- **不注入初速度**（站立 shoot，区别于 crouch-glide）。

### 未列出的继承 reward
上表并非详尽：env 从 velocity 继承了一些低权重的通用 regularizer —— `angular_momentum` (-0.02)、`dof_pos_limits` —— 保留（用于稳定，可忽略）。
⚠️ `soft_landing`（行走 reward）**移除**：它读取被删掉的 2 脚传感器 `feet_ground_contact`（取而代之的是单脚传感器），否则在第 1 个 step 抛 KeyError；且对站立 shoot 无效。

### 传感器重命名 gotcha (⚠️)
重命名脚传感器（`feet_ground_contact` → `left_foot_ground_contact`）会破坏 velocity/ground_pick 继承中所有按此名引用的内容。需要处理：
- **critic obs** `foot_air_time`/`foot_contact`/`foot_contact_forces` → 重新指向左脚传感器（critic 保留支撑信息；否则 env 构建时 KeyError）。
- **reward** `soft_landing` → 移除（见上文；否则第 1 个 step KeyError）。
始终通过 live 构建验证 + **至少跑一个 `step()`**（reward manager 只在 step 时运行），而不仅仅是 cfg 构建或单元测试。

### ⚠️ 学到的重量转移（首次训练后的修订）
观察：在**双手持机器人（双足支撑）**下记录的 BACK/FWD pose 在所有阶段都将 CoM **保持在两脚之间居中**（约左脚内侧 4-5 cm）。在强制 `upright` 下，右脚一抬起机器人就翻倒 → 没有任何 policy 能站住（几何问题，非调参）。已在 sim 中验证（CoM vs 脚 sites）。

采用的修复（RL 自己学平衡）：
- `mdp.com_over_support_foot`：高斯 reward（std 4 cm）将 CoM 投影（`root_com_pos_w`）拉向支撑脚，由 `mdp.kick_engagement` gating（STAND 静止时为 0，踢球时为 1）。权重 3.0。
- **pose 跟踪拆分**（在 `kick_pose_track`/`_l1` 上加 `joint_names` 参数）：
  GESTE = 右腿 + 颈/头（std 0.35，紧）；APPUI = 左腿（std 0.9，权重 1.0，**松**）→ policy 可以内收/平移骨盆来转移重心，而不被跟踪冻结在居中骨盆。
因此上表「平衡/支撑」被扩展：新增 `support_leg_pose` (1.0)、`com_over_support` (3.0)，且 `kick_pose_track`/`kick_pose_l1` 只作用于 gesto 的 9 个关节（右腿+颈）。

### 目标：跟踪按相位插值的 pose
在 `mdp.py` 新增**纯函数**：
```python
kick_pose_target(phase, stand, back, forward, windup_end, kick_end, return_end) -> Tensor
```
按 4 段在归一化周期 [0,1) 上插值 pose 向量：
```
[0, windup_end)        STAND   → BACK      (armement,     défaut 0.35)
[windup_end, kick_end) BACK    → FORWARD   (frappe sèche, défaut 0.10 = "snap")
[kick_end, return_end) FORWARD → STAND     (retour,       défaut 0.30)
[return_end, 1.0)      STAND              (repos)
```
「snap」来自短的踢球段：关节目标快速移动 → 脚快速摆动。3 个 timing 边界可配置。

按**名称**解析关节（`asset.find_joints([name])`）—— 对顺序稳健。

跟踪 reward（始终激活，像 crouch 一样对称）：
| Reward | 权重 | 角色 |
|---|---|---|
| `kick_pose_tracking` | 6.0 | 高斯跟踪 `exp(-((q-cible)/std)²).mean`, std=0.4 |
| `kick_pose_l1` | 2.0 | bootstrap L1（早期梯度恒定） |

### 平衡/支撑（单腿 = 翻倒风险）
| Reward | 权重 | 角色 |
|---|---|---|
| `upright` | 2.0 | 躯干竖直 |
| `support_foot_grounded` (左脚) | 6.0 | 保持支撑脚着地（单脚传感器 → `found∈{0,1}` → reward∈{0,0.5} 除以 2 后，故权重 6.0 ≈ 最大贡献 3.0) |
| `feet_flat` (左) | -1.0 | 左脚板平 |
| `self_collisions` | -1.0 | |
| `body_ang_vel` | -0.05 | |

`support_foot_grounded`：复用 ground_pick 的 `feet_grounded_reward` 机制，但限制在**左脚**（基于 `left_foot_collision` 的 contact 传感器）。

### 正则化（相对 ground_pick 减弱 —— 让 snap 通过）
| Reward | 权重 | 角色 |
|---|---|---|
| `action_rate_l2` | -0.5 | 轻量：过重会杀死快速踢球 |
| `neck_action_rate_l2` | -0.5 | 头部稳定 |
| `joint_torques_l2` | -1e-3 | |

**移除**（行走项）：`track_linear_velocity`, `track_angular_velocity`、`air_time`、`foot_clearance`、`foot_swing_height`、`foot_slip`、`pose`。

### 观测 / 部署（对齐）
- Obs **61D 与 ground_pick/roller 相同**：`[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`，head(4)+body(6) slot **zero-padded**（`zero_command_padding`）。
- 继承自 velocity 的同样 sim2real DR（CoM、mass/inertia、friction BAM、armature、IMU misalignment obs-level、encoder-bias、pushes ±0.3），以 NaN guard 结束。
- 通过现有 export 脚本导出 ONNX（bake normalizer）。
- 部署到 runtime 的 phase slot，例如：
  ```
  --ground-pick shoot.onnx --ground-pick-period 2.5 \
  --ground-pick-kp-ratio 1.0 --ground-pick-action-scale <match>
  ```
  按键 → shoot → 自动返回主 policy。

## 测试

- `tests/test_shoot.py` — 纯函数：
  - `kick_pose_target` 在 keypoints 上：φ=0 处 STAND，`windup_end` 处 BACK，`kick_end` 处 FORWARD，rest 段 STAND；段中点插值；边界（每个分量在 poses 的 min/max 之间）。
  - 简单 case 上的 `kick_pose_tracking` / `kick_pose_l1` reward 值。
- `tests/test_shoot_cfg.py` — env 以正确指令（`GroundPickPhaseCommand`、`randomize_phase=False`、周期）构建，期望 reward 存在 / 行走项缺失。
- 运行：`uv run --with pytest pytest tests/ -q`。

## 训练

```bash
uv run train Mjlab-Shoot-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations <N>
```
监控 `Episode_Reward/kick_pose_tracking`（应上升）。Play：play_latest 脚本。

## 待解决 / 训练时调整
- **Timings**（windup/kick/return）和**周期**：snap 默认值合理，按获得的脚速与稳定性调整。
- **`action_rate` 权重**：snap vs sim2real 平滑的张力；起步轻量（-0.5）。
- **可选增强（v1 未采用）**：在踢球段上 gating 一个小的「右脚向前速度」reward，以在没有真实球的情况下推动功率。仅在纯 pose 跟踪不够有力时添加。
- **部署时过渡**：如果 `STAND_POSE` ≠ 主 policy 的中立 pose，触发/返回时有轻微颠簸（如 crouch 所述）。
