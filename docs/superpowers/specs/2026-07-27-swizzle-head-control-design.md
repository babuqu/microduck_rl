# Swizzle 头部控制（Y 按钮）— 设计

**日期：** 2026-07-27
**分支：** `new_pre_alpha_rollers`
**涉及任务：** `Mjlab-Velocity-Swizzle-MicroDuck` (`microduck_velocity_swizzle_env_cfg.py`)

## 目标

让操作者在鸭子滑行（rollers）时移动鸭子的头部到不同 pose（Y 按钮），**且头部移动时不会让 swizzle 崩掉**。头部 pose 即物理头/颈关节（看上/下/左/右）—— 与 `heading_tracking` 无关（那是躯干的行进方向，此处不动）。

当前 swizzle policy 在头部作为外部偏移移动时会翻倒（它不补偿 CoM 偏移），因此头部必须是**policy 管理的**：policy 产出头部 pose 并保持平衡。这与行走 policy 在 `--new-cmd-obs` 模式下的做法一致 —— 头部是注入到观测中的 COMMAND，由 policy 产出 pose（无外部「双重相加」）。

## 方案（选择：A —— 通过 obs command 让 policy 管理头部）

将 `microduck_velocity_env_cfg.py` 中已有的头部 command 机制移植到 swizzle env：向当前 zero-padded 的 `head_command` obs slot 注入真实头部 pose command，奖励头部跟踪该 command，并通过 curriculum 在后期（LATE）逐步启用，以避免干扰 swizzle。

不采用外部偏移方案（先前已否决：若头部移动而 policy 不补偿，swizzle 无法保持直立）。

## 变更（全部在 `make_microduck_velocity_swizzle_env_cfg` 中）

1. **头部 pose command 项。** 添加 `cfg.commands["head_pose"] = UniformPoseCommandCfg(...)`，从 velocity env 复制：4D `[neck_pitch, head_pitch, head_yaw, head_roll]` 相对默认的偏移，`resampling_time_range = (2.0, 5.0)`，每关节 range（head_roll 更窄，匹配其较小的机械行程）。
2. **真实 `head_command` obs。** 将当前 `zero_command_padding(dim=4)` 的 head slot 替换为真实 command obs `func=<head command obs>, params={"command_name": "head_pose"}`，actor 和 critic 都要改。（保持 61D 布局；body_command 仍 zero-padded —— 此处不做 body-pose 控制。）
3. **`head_pose_tracking` reward。** 添加 `cfg.rewards["head_pose_tracking"]`
   (`microduck_mdp.head_pose_tracking`, `command_name="head_pose"`, `std=0.5`)，初始权重 0（由 curriculum 逐步启用）。
4. **后期 curriculum。** 一个 `reward_weight` curriculum 将 `head_pose_tracking` 从 0
   → **4.0**，在 **~1500 iter**（swizzle 已稳定）之前保持 0，随后在接下来 ~1000 iter 内爬升，对齐 velstand 的 body-pose 启用节奏。再加一个头部 pose command range curriculum：起始用紧 range（小幅头部偏移），在同一窗口内逐步扩大，使早期头部几乎不动，等 policy 能处理时再达到全 range。这正是让头部控制「不难管」的关键 —— 它叠加在已稳定的 swizzle 之上。（数值为起点，可调。）
5. **协调颈部惩罚（必需）。** env 当前有 `neck_joint_pos_l2`，将颈/头关节拉向 HOME —— 它会与 `head_pose_tracking`（拉向 command）冲突，导致头部永远不动。需将头部 pose 关节从 `neck_joint_pos_l2` 排除（或直接删除），对齐 velocity env 的做法（其注释：两者并存会"一边拉向 HOME 一边 head_pose_tracking 拉向 command"）。保留 `neck_action_rate_l2`（平滑性，无冲突）。

其余一切（swizzle、向后行走、heading curriculum、DR、obs 布局、command）不变。需要**重新训练** swizzle 任务。

## Runtime

无需改 runtime 代码。`microduck_runtime` 的 **Y 按钮** 已经驱动 `head_command` obs slot（new-cmd-obs 模式把头部偏移作为 command 注入，"不要双重相加"）。swizzle policy 重新训练带头部控制后，Y 即可响应。
部署 flag 不变（`--roller --new-cmd-obs ...`）。

## 测试 / 验证

- Smoke 测试：`uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16
  --agent.max_iterations 2` 能跑；`head_pose_tracking` 出现在 reward log 中；`head_pose`
  command 与真实 `head_command` obs 构建无报错。
- 正式训练：`head_pose_tracking` 在 curriculum 启用后上升；swizzle 保持稳定（头部
  curriculum 开启时摔倒率不飙升）。在 viewer / 真机上：移动头部 command 会移动头部，
  roller 继续滑行。

## 调参旋钮

- 头部在启用时扰乱 swizzle → 推迟 curriculum 启用时机，或更慢地扩大头部 range。
- 头部跟踪不佳 → 提高 `head_pose_tracking` 目标权重，或检查颈部惩罚是否仍在干扰。
- 头部抖动过大 → 保留/提高 `neck_action_rate_l2`。
