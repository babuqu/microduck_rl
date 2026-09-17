# Swizzle 滑轮环境 — 设计

**日期：** 2026-07-23
**分支：** `new_pre_alpha_rollers`

## 目标

一个**独立的**滑轮任务，产生**干净的经典 swizzle**：两个刀片都保持在地面上，双腿**对称地**向外张开再收回（沙漏形态），推动鸭子前进。这是交替跨步（`Mjlab-Velocity-Flat-MicroDuck-Rollers`）的一个更简单、更稳定的替代方案，动机是跨步动作在真实机器人上迁移效果不佳。跨步环境保持不变。

Sim2real 是一个目标：与跨步环境使用相同的机器人、观测、命令语义、domain randomization 和 ONNX 导出，因此它**以相同方式部署**（`microduck_runtime ... --roller`，相同的 flag）。

## 方法（已选：A — 移除 anti-swizzle + reward 对称性）

基础滑轮 velocity recipe *自然地*收敛到 swizzle（这是我们在跨步中对抗的吸引子）。因此获得干净 swizzle 最简单的方式是**移除 anti-swizzle 机制**并**奖励 swizzle 的定义性特征**（对称性、足部着地）。无 phase scripting。

已拒绝：B（显式沙漏足部模式塑造）和 C（phase 驱动的脚本化轨迹）— 更复杂，仅在 A 的 swizzle 看起来凌乱（节奏/幅度）时才需要。

## 结构

- 新文件 `src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py`，包含 `make_microduck_velocity_swizzle_env_cfg(play=False)` 和 `MicroduckSwizzleRlCfg`。基于 `make_velocity_env_cfg()` + 滑轮机器人构建，镜像 `microduck_velocity_rollers_env_cfg.py` 的结构（obs、DR、command、curricula）。
- 在 `tasks/__init__.py` 中注册 `Mjlab-Velocity-Swizzle-MicroDuck`。
- 复用跨步环境中所有 sim2real 相关内容：机器人 cfg、61D obs 布局、command（cmd_x push/coast/brake，直线：`ang_vel_z=(0,0)`，`heading_hold`）、所有 DR 事件 + curricula（com、wheel_friction）、`action_over_limit`、ONNX 导出路径。

## Reward recipe

**保留**（任务 + 稳定性 + sim2real）：
`wheel_speed`（前进推进，任务）、`braking`、`upright`、`com_height_target`、`pose`、`forward_lean`、`heading_hold`、`action_over_limit`、`feet_flat`、`self_collisions`、regularizer（`action_rate_l2` + curriculum、`neck_action_rate_l2`、`neck_joint_pos_l2`、`joint_torques_l2`）。

**移除**（跨步 / anti-swizzle 机制）：
`single_support`、`glide`、`skating_air_time`、`gait_symmetry`、`hip_roll_neutral`（最后一项会与 swizzle 的横向外向运动相冲突）。

**新增**（pro-swizzle）：
- `leg_symmetry` — 奖励左右腿镜像。机器人使用镜像的 L/R 符号约定，因此对称配置满足每对 `q_left + q_right ≈ 0`。返回 `-mean_pairs |q_left + q_right|`（L1，恒定梯度 — 与现有 `bilateral_symmetry_penalty` 相同形式），覆盖腿关节对（hip_yaw、hip_roll、hip_pitch、knee、ankle）；与正权重一起使用以惩罚不对称并偏好对称的 swizzle。这是 swizzle 的定义性特征。（实现：现有 `bilateral_symmetry_penalty` 接受显式的 L/R 索引列表；添加一个薄包装器，在运行时按名称解析 L/R 腿关节对，以便无需硬编码索引即可配置。）
- `grounded` — 奖励两个刀片同时接触（n_contact == 2）时推动，使足部保持向下（经典 swizzle，无抬起）。小权重。新的 mdp 函数（`single_support_reward` 的镜像，但奖励双支撑）。像其他一样在 `cmd_x >= 0` 上 gate。

保持 `hip_roll` pose std 宽松（如跨步环境），以便腿可以张开。

## 新的 mdp 函数（在 `tasks/mdp.py` 中）

1. `leg_symmetry_reward(env, asset_cfg)` — 按名称解析 L/R 腿关节对，返回 `-mean_pairs |q_left + q_right|`（与正权重一起使用）。
2. `grounded_reward(env, sensor_name, command_name)` — 奖励恰好两个刀片接触，按 `clamp(cmd_x, 0)` 缩放。

## Command / sim2real（与跨步相同）

`cmd_x` push/coast/brake，`lin_vel_y=0`，`ang_vel_z=(0,0)`（直线），完整 DR（com、head_com、mass/inertia、joint friction、armature、wheel friction、velocity pushes、IMU misalignment、encoder bias、obs delays），61D obs，`vel_scale=0.3`。以与跨步滑轮 policy 相同的 runtime flag 部署。

## PPO 配置

复用 `MicroduckRollersRlCfg` 的超参数（相同的 actor/critic 512-256-128 ELU、PPO 设置、`entropy_coef=0.03`），新的 `experiment_name`/`run_name` = `velocity_swizzle`。

## 测试 / 验证

- Smoke test：`uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 2` 无错误运行；`leg_symmetry` 和 `grounded` 出现在 reward 日志中。
- 在真实运行中观察：`leg_symmetry` 高（对称），`grounded` 高（双足着地），`wheel_speed` 上升（向前移动）。视频：对称的沙漏 swizzle，两个刀片都在地面上。

## 调节旋钮（首次运行后）

- 如果不够对称 → 提高 `leg_symmetry` 权重。
- 如果抬起脚 → 提高 `grounded` 权重。
- 如果几乎不移动 → 对称/着地权重相对于 `wheel_speed` 过高；降低它们。
- 如果 swizzle 看起来凌乱（节奏/幅度）→ 升级到方法 B。
