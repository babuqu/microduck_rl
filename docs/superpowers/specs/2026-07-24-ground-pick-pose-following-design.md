# Ground-pick 通过 phase 插值的 pose 跟随

**日期：** 2026-07-24
**分支：** `new_pre_alpha_ground_pick`
**目标文件：** `src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py`（原地重写）
**Task id：** `Mjlab-GroundPick-Flat-MicroDuck`（不变）

## 1. 目标

用 **pose 跟随指令性目标**替换当前 ground_pick 的 *任务空间* 目标（奖励嘴巴下降到地面，然后单独奖励返回站立）：定义两个目标关节 pose — STAND 和 DOWN — 并奖励对 **phase 插值 pose**（STAND→DOWN→STAND）的跟随。

动机（沿用经验证的 roller_crouch 方法）：插值 pose 目标 **构造上对称** — « 起身 »（目标 → STAND）与 « 下降 »（目标 → DOWN）获得完全相同的奖励，这解决了 policy 下降但起身不佳的懒惰最优问题。信号在**每个 phase 都是密集的**（连续移动的目标），不同于由 `sin` 加权的固定目标在转换处不提供任何信号。

手势仍然通过 runtime 的 `--ground-pick` slot 由 **A 按钮** 触发（一次性，自动返回主 policy）。统一的 61D obs 不变 → policy 在 slot 中可互换。

## 2. 目标 pose

**按名称**解析关节（`asset.find_joints([name])`）— 稳健，与 roller 方法一致。14 个关节（排除 mouth）。

- **STAND_POSE** = HOME（模型的 `default_joint_pos`）。blend 的源；不要硬性重新定义 — 使用模型默认值作为源（blend=0）。部署时，主 policy 从 HOME 恢复 → 干净的返回。

- **DOWN_POSE** = 来自 `scene_walk.xml` 的 **keyframe FOLD** 的初始值（深度前折，头部低下 → 嘴朝向地面）。文件头部的按名称 dict，**注释为可用 `read_pose.py` 读取真实机器人嘴朝下放置的 pose 来替换**。起始值：

  ```python
  DOWN_POSE = {
      "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 1.57,
      "left_knee": 1.57, "left_ankle": 0.0,
      "neck_pitch": 1.0, "head_pitch": 1.0, "head_yaw": 0.0, "head_roll": 0.0,
      "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.57,
      "right_knee": -1.57, "right_ankle": 0.0,
  }
  ```

## 3. Phase 轮廓（4 段）

命令 `GroundPickPhaseCommand`：`[cos(2πφ), sin(2πφ), 0]`，周期 **4.0 s**（runtime slot 默认值 → 部署时无需更改 period flag）。

```
DESCENT_END=0.15  HOLD_END=0.50  RISE_END=0.65   (période 4 s)
[0, 0.15)     descente  STAND->DOWN   ~0.6 s   blend 0->1
[0.15, 0.50)  bas       DOWN          ~1.4 s   blend 1
[0.50, 0.65)  remontée  DOWN->STAND   ~0.6 s   blend 1->0
[0.65, 1.0)   haut      STAND (repos) ~1.4 s   blend 0
```

`blend ∈ [0,1]`：0 = STAND（HOME），1 = DOWN。目标 = `stand + blend·(down - stand)`。可调边界（文件头部的常量）。

**`randomize_phase=False`**：每个 episode 从 φ=0（= 站立）开始，如同部署时的 A 按钮触发。episode 在错开的时间重置，env 在 phase 上自然解相关（无需 randomize）。需要向 `GroundPickPhaseCommandCfg` 添加 `randomize_phase` flag（默认 `True` → 其他 sit/stand 任务不变），在 `reset()` 中遵循。

## 4. 新的 mdp 函数（来自 roller，按名称适配）

在 `src/mjlab_microduck/tasks/mdp.py` 中。名称与现有 `phase_pose_match`（即 sin 加权固定目标变体）不同，以避免混淆。

- **`phase_pose_blend(phase, descent_end, hold_end, rise_end) -> Tensor`** — 纯函数，4 段 blend 0..1（可隔离测试）。
- **`_phase_pose_error(env, asset_cfg, command_name, target_pose, descent_end, hold_end, rise_end, source_pose=None) -> (cur, target)`** — 按名称解析关节；`source_pose` = HOME（`default_joint_pos`）如果为 `None`；计算 `phase = atan2(sin,cos)/2π % 1`、`blend`，然后 `target = source + blend·(target_pose - source)`。
- **`phase_pose_track(env, command_name, target_pose, source_pose=None, std=0.3, descent_end, hold_end, rise_end, asset_cfg) -> Tensor`** — 高斯 `exp(-((cur-target)/std)²).mean(-1)`。
- **`phase_pose_track_l1(env, ...相同 args 无 std...) -> Tensor`** — bootstrap `-(cur-target).abs().mean(-1)`（高斯饱和时恒定梯度）。

`target_pose` = `DOWN_POSE`（按名称 dict）。`source_pose=None` → HOME。

## 5. Rewards

相对于当前的最小化重写 — 替换 pose 返回机制，保留稳定性/regularizer/sim2real。

| Reward | 权重 | 状态 | 角色 |
|---|---|---|---|
| `phase_pose_track` (std 0.3) | **6.0** | **新增** | 跟随插值 pose STAND↔DOWN |
| `phase_pose_track_l1` | **2.0** | **新增** | bootstrap L1 |
| `mouth_ground_proximity` (std 0.10) | **1.0** | retune（原为 2.0） | 安全网：如果 DOWN 不完美则保证嘴着地；在接近上 gate（+sin） |
| `upright` | 0.2 | 保留 | 躯干 ~垂直（低，机器人倾斜） |
| `feet_grounded` | 3.0 | 保留 | 整个手势期间 2 脚着地 |
| `self_collisions` | -1.0 | 保留 | |
| `head_impact_penalty` (阈值 2 N) | -0.5 | 保留 | 无头部 slam（DOWN 使头部降低） |
| `action_rate_l2` | -0.8→-2.0 (curric) | 保留 | 平滑 |
| `neck_action_rate_l2` | -1.0 | 保留 | |
| `joint_torques_l2` | -5e-3 | 保留 | |
| `body_ang_vel` | -0.05 | 保留 | |
| `angular_momentum` | -0.02 | 保留 | |
| `soft_landing` | -1e-5 | 保留 | |

**移除**：`mouth_perpendicular_to_ground`、`ground_pick_return_pose_legs`、`ground_pick_return_pose_neck`（被 pose 跟随取代）。

其余全部**不变**：DR 块（CoM/head-CoM/mass-inertia/friction/armature/IMU-misalign/encoder-bias/pushes），obs 61D + padding head/body 零，终止（`nan_state`），curricula（`action_rate_weight`、`com_range`、`head_com_range`），RlCfg（`experiment_name="ground_pick"`）。

## 6. 部署（sim2real 对等）

```bash
microduck_runtime ... \
  --ground-pick ground_pick.onnx \
  --ground-pick-period 4.0 \       # = période env (défaut, rien à changer)
  --ground-pick-kp-ratio 1.0 \     # entraîné kp 200 → forcer 1.0 (défaut 0.6 baisse à 120)
  --ground-pick-action-scale 1.0   # = action.scale env
```

## 7. 测试

`tests/`（运行 `uv run --with pytest pytest tests/ -q`）：

- **纯函数**：`phase_pose_blend` 在关键点（φ=0→0、φ=0.075→0.5、φ=0.3→1、φ=0.575→0.5、φ=0.8→0、每段单调）；`phase_pose_track`/`_l1`：最大值（cur==target）和符号。
- **env 构建**：`make_microduck_ground_pick_env_cfg()` 构建；命令 = `GroundPickPhaseCommand`，`randomize_phase=False`、`period=4.0`；reward `phase_pose_track`/`phase_pose_track_l1` 存在；`mouth_perpendicular_to_ground`/`ground_pick_return_pose_*` 不存在；`mouth_ground_proximity` 存在权重 1.0。

## 8. 训练 / play / 导出

```bash
uv run train Mjlab-GroundPick-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 20000
uv run scripts/play_latest.py     # md-play
uv run scripts/export_latest.py   # normaliseur baké dans l'ONNX
```
监视 `Episode_Reward/phase_pose_track`（应上升）。

## 9. 范围外 / 注释

- **重复 `pose_target_match`**（mdp.py 1577 和 1914）：潜在的，此处不处理。
- **DOWN_POSE 调整**：如果嘴用 FOLD 值不够接触地面，调整 dict（理想情况下用 `read_pose.py` 读取真实机器人嘴朝下放置的 pose）而不是增大 `mouth_ground_proximity`。
- **部署转换**：STAND=HOME = 主 policy 的中性点 → 返回时无颠簸（不同于 roller 上注意到的 STAND≠HOME 的担忧）。
