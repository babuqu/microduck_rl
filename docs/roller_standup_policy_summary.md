# Policy `roller_standup` — 穿着轮滑站起来

**目标**：microduck（穿着轮滑）从地面开始——俯卧或仰卧——并**重新站立到轮子上**，然后**保持**站姿。

- **任务**：`Mjlab-RollerStandUp-Flat-MicroDuck`
- **文件**：`src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- **基础**：衍生自 roller env（`velocity_rollers`）→ 同一个机器人、同一个物理/DR、**同一个 61D observation**（runtime 可互换，可通过 `--new-cmd-obs` 加载）。
- **规格**：`docs/superpowers/specs/2026-08-04-roller-standup-design.md`
- **盲 policy**：不扫描地形；仅本体感受 + `projected_gravity`。

## 高度（测量值，非猜测）

| 姿势 | 脚模型 | 轮滑模型 |
|---|---|---|
| 站立 | 0.1172 → 负载下 `STAND_Z=0.115` | 0.1407 → **`ROLLER_STAND_Z=0.138`** |
| 俯卧（静止） | 0.075 | 0.075 |
| 仰卧（静止） | 0.048 | 0.048 |

地面静止高度在两个模型上相同：因为接触地面的是躯干外壳，而不是脚。

## ⚠️ 关节索引——轮子是交错排列的

```
0-4   左腿      5-6   左轮
7-10  颈/头     11-15  右腿      16-17  右轮
```
`_LEG_JOINTS = [0-4, 11-15]`。`standup` 中的索引（`[0-4, 9-13]`）只适用于**无轮**模型，在这里会指向轮子。由 `tests/test_roller_standup_cfg.py::test_joint_indices_match_actual_roller_model` 锁定。

## Reset——从地面开始

`set_random_ground_state`：俯卧（`prone_z` 0.076–0.09，下限抬高，因为俯卧姿势只有在 0.0752 时才离开地面）/ 仰卧 / **已经站立**（`standing_z` 0.134–0.144），± 10° 的 pitch/roll 噪声。没有"坐着"桶。"站立"桶是必需的：没有它，policy 能起来但站不住。

**Curriculum `ground_state_mix`**（易 → 难，仰卧在最后）：

| iter | 站立 | 俯卧 | 仰卧 |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

## Reward

从 `standup` 继承的十个 term 及其已调好的权重：`pose_stand_legs` (+8)、`pose_stand_l1` (+5)、`height_stand` (+4, std 0.04)、`height_stand_sharp` (+4, std 0.015)、`height_stand_l1` (+30)、`com_upward_velocity` (+3)、`gentle_rise` (−0.02)、`upright_linear` (+6)、`upright_sharp` (+6)、`standing_composite` (+15)。加上 `joint_torque_rate_l2` (−2e-3)，一个不阻止翻转的 anti-jitter 项。

继承的 regularizer：`body_ang_vel` **−0.05**（motion-blocker，保持轻度）、`angular_momentum` −0.02、`action_rate_l2`（ramp −0.4 → −1.0，**不是** roller 的 −2.0）、`neck_action_rate_l2` −0.5、`neck_joint_pos_l2` −0.5（头部正位）、`joint_torques_l2` −1e-3、`action_over_limit` −0.5、`self_collisions` −1.0。

已移除：所有滑行 reward，以及 `feet_flat`（上升过程中刀片不是平的）和 `hip_roll_neutral`（站起来需要分开腿）。

## ⚠️ 难点：轮子会滚动

没有纵向附着力来推地面。**滚动摩擦 curriculum 是反向的**（roller env 让它上升，这里让它下降）：

| iter | frictionloss | |
|---|---|---|
| 0 | 0.05 | 轮子几乎锁死 → 像有脚一样站起来 |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | 真实的滚动值 |

**在各阶段监视 `Episode_Reward/standing_composite`。**如果它崩塌，"有附着的脚"的动作不能迁移到自由轮上 → 需要引导一种滑冰者技术（中间膝盖支撑，一次一个滑板）。这是一个结果，而不是失败。

**还要在 play 中监视机器人的水平漂移**，在每个摩擦阶段。`standing_composite` 既看不到 `root_link_pos_w[:2]` 也看不到水平速度：一个滑动着远离起点站起来的 policy 与一个站起来并停住的 policy 得到完全相同的分数。只要这个漂移没有经过视觉测量，摩擦 curriculum 的结果（这个 env 存在的目的就是要回答的问题）就是不可靠的。

**Sim2real**：只有 iter 4000 之后的 checkpoint 才是部署候选。在此之前，policy 依赖于真实机器人上不存在的摩擦。

## 命令

`twist` slot 中和：`lin_vel_x`/`lin_vel_y` ± 0.01，`ang_vel_z` **± 0.05**（宽 5 倍——与 `standup` 同样的选择）。`head_pose` / `body_pose` slot **zero-padded**（roller 约定）。目标部署：在 `--standing` 模式下面对 `--walking` roller policy，通过命令的 magnitude 自动切换（`infer_policy.py:262`，阈值 0.05）；那里的 twist slot 保持为零（`infer_policy.py:239`）。

**保留**：`infer_policy.py` 是本地 sim/keyboard 脚本。机器人 runtime 是 Rust 二进制 `microduck_runtime`，不在 repo 中——没有验证它是否暴露了等价的 `--standing`。crouch 的交接文档只列出了 `--model`、`--ground-pick`、`--fold-policy`。待确认。

## 终止

`fell_over` **删除**（机器人从摔倒开始）。继承的 `nan_state`。在 actor/critic obs 上 `nan_policy="sanitize"`。

## 网络 / PPO

Actor 和 critic `(512, 256, 128)` elu，`obs_normalization=True`。PPO `lr=1e-3` adaptive，`desired_kl=0.01`，`gamma=0.99`，`lam=0.95`，`num_steps_per_env=24`，episode 6 s，`max_iterations=15000`。**Symmetry OFF**（`SYMMETRY_CFG` 是为 51D layout 硬连线的）。

## 命令

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
uv run scripts/play_latest.py        # alias md-play
uv run scripts/export_latest.py      # alias md-export
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```

### ⚠️ 在 play 中查看仰卧起始

play 默认**从不**显示仰卧起始：play env 是从头重建的，所以 `common_step_counter` 重置为 0，curriculum 应用其阶段 0，其中 `face_up_prob = 0`。无论加载的 checkpoint 多成熟，只能看到 50% 俯卧 / 50% 站立。而仰卧是最难的情况，正是我们想要检查的。

`STANDUP_PLAY_FACE_UP` 强制混合（与 `roller_slope` 中的 `SLOPE_PLAY_DIFFICULTY` 同样的模式），**仅在 `play=True` 路径上**——训练及其易→难 curriculum 不受影响：

```bash
STANDUP_PLAY_FACE_UP=1.0 md-play    # 100% 仰卧起始
STANDUP_PLAY_FACE_UP=0.4 md-play    # curriculum 最后阶段的混合
STANDUP_PLAY_FACE_UP=none md-play   # 默认（阶段 0，没有仰卧）
```

其余（`1 - face_up`）按照最后阶段的 2:1 比例分配给俯卧:站立，因此 `0.4` 精确再现了训练结束时的混合（0.40 / 0.20 / 0.40）。

## 🔧 反暴力修正（在首次机器人测试之后）

**症状** 在 4000+ checkpoint 上：动作非常突兀，头撞地，机器人上从仰卧站起失败。**在仿真中也存在** → 所以既不是 sim2real 问题，也不是 checkpoint 太年轻，而是 reward 设计问题。

**根因：`gentle_rise` 奖励了暴力。**`trunk_vertical_accel_penalty` 已经返回 `-|a_z|`（`mdp.py:2171`）；乘以从 `standup` 继承的权重 **−0.02**，就成了双重否定，即 `+0.02·|a_z|`——**躯干加速越猛烈，policy 被支付得越多**。日志确认：run `vweolw91` 上 `Episode_Reward/gentle_rise = +0.0118`，是唯一一个被记录为正值的 penalty term。

`mdp.py` 混合了两种 sign convention，这就是陷阱：

| term | 函数返回 | 正确权重 |
|---|---|---|
| `height_stand_l1`、`pose_stand_l1`、`gentle_rise` | `-abs(...)`，已经为负 | **正** |
| `joint_torques_l2`、`joint_torque_rate_l2`、`action_rate_l2`、`body_impact_cost` | 正的 magnitude | **负** |

由 `test_already_negative_penalties_use_positive_weights` 锁定。

⚠️ **步行者的 `standup` 有完全相同的 bug**（同一函数，同一权重 −0.02）。这解释了其注释中记录的一系列失败的阻尼尝试（"*violent / shaky / overshoot-tip-repeat on the real robot*"）：它们在对抗一个主动推向反方向的 term。**此处未修正**——这是另一个 env，需要单独处理。

**关联的结构性问题。**收敛时，任务 reward 合计达到 **≈ +41.6**，95–99% 饱和，而所有阻尼器合计 **≈ −1.2**——其中 `joint_torque_rate_l2` 为 **−0.0002/step**，`joint_torques_l2` 为 **−0.0001/step**，等于没有。比例约 35:1：没有任何理由要柔和。

**当前修正状态：**

| | 之前 | 现在 | 原因 |
|---|---|---|---|
| `gentle_rise` | −0.02（reward） | **+0.02**（penalty） | 符号修正；magnitude 故意保持小——翻转过程中 `\|a_z\|` 必然很高，大权重会成为 motion-blocker |
| `joint_torque_rate_l2` | −2e-3 | **−0.2** | 安全的杠杆：惩罚 torque 变化，而不是运动 |
| `head_impact_penalty` | 缺失 | **仍然缺失** | 试过 −1.0，冻结了 policy——见下文 |

### ⚠️ 头部 impact penalty 冻结了 policy——不要原样恢复

尝试使用 `velstand` 的值（`body_impact_cost`，`neck` subtree，−1.0，阈值 2.0）：**policy 收敛于保持躺下、不动。**测量结果（run `d8rnko6p`）：

| term | 之前（暴力） | 有 head_impact（冻结） |
|---|---|---|
| `standing_composite` | +14.32 | **+3.26** |
| `upright_sharp` | +5.76 | +1.06 |
| `head_impact_penalty` | — | **−1.01** ← 最大的负 term |
| `joint_torque_rate_l2` | −0.0002 | −0.255（所以**不是**罪魁祸首） |

推理错误：认为一个"有针对性"的 penalty 不会限制运动。**这里错了——从仰卧站起来，这个机器人在它的头和肩膀上旋转。**头是翻转的支点，而不是附带伤害；惩罚它会阻断唯一可用的机制，而仰卧本来就是失败的情况。

**使这种冻结成为可能的懒惰最优**：`pose_stand_legs` 保持 **+7.72/8**，而机器人躺着——腿在躺下姿势下处于 HOME 位置，所以这个 reward 几乎是免费获得的。应该是 `height_stand_l1`（权重 +30）让"留在地面"净负；不应削弱它。

**正在测试的假设**：头撞地是暴力的*症状*（sign bug 奖励了猛烈，猛烈的上升最终落在头上），而不是一个独立的缺陷。如果现在 sign 修正后 slam 回来了，恢复应该是一个**高度门控**的 penalty（像 `upright_sharp` 那样），在地面翻转阶段豁免。

**方法论教训**：三个修正是一次性应用的，所以冻结无法确定性地归因——只能指出最可能的嫌疑。今后一次只改一个。

**如果仍然暴力的重新校准**：`|Δτ|²` 收敛时约 0.1，所以 `joint_torque_rate_l2` 的贡献 ≈ `0.1 × |weight|`。提高**这个** term，**不要**提高 `body_ang_vel`（−0.05）或 `action_rate_l2`（ramp → −1.0）：那些是 motion-blocker，`standup` 文档记录在 −0.15 和 −1.2 时它们**冻结**了从仰卧站起。相反，如果仰卧停止工作，**首先降低** `joint_torque_rate_l2`。

## 超出范围

将站起集成到滑行 policy 中（`velstand` 配方）；侧躺起始桶；rough 变体；躯干/头部 impact penalty。

没有任何 reward 惩罚躯干水平速度（`root_link_lin_vel_w[:, :2]`）："滑着远离站起来"是一个未被惩罚且拿满分的结果。故意决定（不是疏忽）：一个不以高度门控的静止 reward 也会惩罚从地面站起所物理上必需的平移——`standup` 文档记录的"motion-blocker"失败模式。如果问题被确认的候选：一个以高度门控的静止（仅在接近 `ROLLER_STAND_Z` 时）。
