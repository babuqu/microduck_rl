# Design — `roller_standup`：穿轮滑鞋重新站立

**目标**：一个专门的 policy，让 microduck 在摔倒后（趴着或仰躺着）**重新在轮子上站起来**，
并能随后**保持**轮上站立姿势。

将 `standup`（行走鸭子）的配方移植到 rollers 模型。不修改任何现有 env。

---

## 已定决策

| 决策 | 选择 | 被排除的备选 |
|---|---|---|
| 形式 | **专门的 episodic** policy | 将站立动作嫁接到 roller env 上（`velstand` 配方）→ 有实际破坏已学步态的风险 |
| 起始姿势 | **趴着 + 仰躺 + 站立** | `坐姿`（只存在于从 `sit` policy 切换的场景，roller 无对应）；侧躺（覆盖最广但收敛难度大得多）；不含 `站立`（policy 会站起来然后又摔倒） |
| 自由轮 | **反向滚动摩擦 curriculum** | 使用真实输入摩擦（bootstrap 太难）；通过 reward 强加滑冰技术（仓库历史：过于直接的 style reward 会产生寄生最优解 — swizzle、crouch 的懒惰最优） |
| 目标姿势 | **HOME + 实测高度** | roller-crouch 的 `STAND_POSE`（被标记为未决问题：与 roller 中性姿势不一致 → 返回时有冲击）；从真实机器人读取的姿势（会阻塞开发） |
| 命令 | **twist 中性化**（≈ 0） | 相位 / 按钮槽命令（见「部署」）；可控制的头部 |

---

## 架构

**新文件**：`src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- `make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`
- `MicroduckRollerStandUpRlCfg`（`experiment_name="roller_standup"`）
- Task id：`Mjlab-RollerStandUp-Flat-MicroDuck`（仅 flat，无 rough 变体）

**派生方式**：`cfg = make_microduck_velocity_rollers_env_cfg()`。

这是 `roller_slope`（246 行）的模式，而不是 `roller_crouch`（479 行，从
`make_velocity_env_cfg()` 重新开始并复制所有 DR 块）的模式。这样继承可以无漂移地获得：

- robot `MICRODUCK_WALK_ROLLERS_ROBOT_CFG`（14 个主动关节 + 4 个被动轮，BAM m6，kp_fw 200）；
- 传感器 `feet_ground_contact`（`ankle_{l,r}_v1` 上的 subtree 模式）和 `self_collision`；
- 全部 DR：躯干 + 头部 CoM、质量/惯性（pseudo_inertia）、BAM 摩擦、armature、
  编码器偏置、obs 级 IMU 失准、轴承摩擦；
- **统一的 61D observation** `[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14),
  last_action(14), command(13)]` — 这是 runtime 可互换性的硬条件；
- `nan_state` 终止（扩展守卫：关节 + free-joint + 轮子）。

rollers 模型**在物理上允许**躺下：`robot_allcollisions_rollers.xml` 在躯干
（`np_f970`）、髋部、腿部、头壳和下巴上都有碰撞几何体，
还有 4 个轮胎。已验证。

---

## 实测常量

通过精确运动学测量（碰撞几何体的网格顶点最小值、keyframe 的 `STAND` 姿势、躯干触地）
在 `scene_rollers.xml` 与 `scene.xml` 上对比：

| 姿势 | 脚模型 | rollers 模型 |
|---|---|---|
| 站立（`STAND` = HOME） | 0.1172 | **0.1407** |
| 趴着（静止） | 0.0752 | 0.0752 |
| 仰躺（静止） | 0.0476 | 0.0475 |

一致性校验：`standup` 使用**受载下**测得的 `STAND_Z = 0.115` 对比运动学的 0.1172 →
约 2 mm 下沉。我们应用同样的修正，结果恰好落在 roller env 已使用的
`reset_base z = 0.1335–0.1435` 范围内。

```python
ROLLER_STAND_Z   = 0.138   # tronc debout sur roues, sous charge (+23 mm vs pieds)
ROLLER_PRONE_Z   = 0.075   # hauteur de repos à plat ventre
EPISODE_LENGTH_S = 6.0
```

两个模型在地面上的静止高度**完全相同**（因为触地的是躯干外壳，不是脚）。
但这并不意味着 `standup` 的 `prone_z` 范围可以直接套用：见「Reset」下的注释 —
`prone_z_min` 不同（此处为 0.076，不是 0.05），因为一个范围同时服务两个姿势
（趴、仰），它们在 reset 时的触地高度并不相同。

测得的量正是 reward 所读取的：`height_target_gaussian` 和
`height_l1_penalty` 使用 `root_link_pos_w[:, 2]`，其值恰好等于 `xpos[trunk_base].z`
（free-joint 在 `trunk_base` 上）— 数值已验证。

## 关节索引

被动轮**交错插入**关节顺序。在 MuJoCo 中验证的实际顺序
（`m.jnt_qposadr`，rollers 模型，free-joint 后 18 个关节）：

```
0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
5-6   passive_LF_wheel, passive_LR_wheel
7-10  neck_pitch, head_pitch, head_yaw, head_roll
11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
16-17 passive_RF_wheel, passive_RR_wheel
```

```python
_LEG_JOINTS   = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]   # standup : [0-4, 9-13]
_NECK_JOINTS  = [7, 8, 9, 10]                          # standup : [5-8]
_WHEEL_JOINTS = [5, 6, 16, 17]
```

只有 `_LEG_JOINTS` 被真正消费（用于姿势 reward）。`_NECK_JOINTS` 和
`_WHEEL_JOINTS` 仅为文档和索引测试而声明：颈部通过**名称**解析
（`neck_joint_pos_l2` 在每步调用 `find_joints(r".*(neck|head).*")`，正是为了
对轮子造成的偏移保持稳健），轮子通过正则 `^passive_.*` 解析。

交接文档明确标注了这一脆弱点。由一个构建 env 并核对这些索引关节名的测试锁定
（见「Tests」）。

---

## Reward

### 从 roller 继承中移除

| 移除 | 原因 |
|---|---|
| `wheel_speed`、`braking`、`skating_air_time`、`glide`、`single_support`、`gait_symmetry`、`forward_lean`、`heading_hold` | 步态 reward：趴在地上时毫无意义 |
| `feet_flat` | 站起过程中轮 blade 不会平贴 → 这个惩罚会对抗该动作 |
| `hip_roll_neutral` | 站起来需要分开腿 |
| `pose`、`com_height_target` | 被下方的姿势/高度目标替代 |
| `upright`（基础高斯） | 被 `upright_linear` + `upright_sharp` 替代 |

### 从 roller 继承中保留

| Reward | 权重 | 作用 |
|---|---|---|
| `action_over_limit` | −0.5 | sim2real 保护（超出挡块的超程控制），与任务无关 |
| `self_collisions` | −1.0 | |
| `body_ang_vel` | **−0.05** | 故意**轻**：`standup` 记录在 −0.15 时会冻结站起（motion blocker） |
| `angular_momentum` | −0.02 | |
| `action_rate_l2` | curriculum −0.4 → −0.8 → −1.0 | roller env 把它固定在 −1.0；我们采用 `standup` 的渐升（前期柔和 → 帮助大翻身动作的 bootstrap） |
| `neck_action_rate_l2` | −0.5 | 头部稳定 |
| `neck_joint_pos_l2` | −0.5 | 保持头部正中（`roller_slope` 的选择）— **替代** `standup` 的 `head_pose` 命令 |
| `joint_torques_l2` | −1e-3 | |

### 新增

| Reward | 权重 | 作用 |
|---|---|---|
| `joint_torque_rate_l2` | −2e-3 | anti-jitter：`standup` 将其识别为唯一不阻塞翻身的阻尼器（它惩罚扭矩的*变化*，而非幅值，也不惩罚躯干旋转） |

### 站立 reward（从 `standup` 移植，重新映射）

十个术语从 `microduck_standup_env_cfg.py` 中记录的迭代**带着已调好的权重**复制。
仅关节索引和两个高度不同。所有 mdp 函数已存在 — **`mdp.py` 无需写任何东西**。

| Reward | mdp 函数 | 权重 | roller 参数 | 作用 |
|---|---|---|---|---|
| `pose_stand_legs` | `pose_target_match` | +8.0 | `std=0.5`、`joint_indices=_LEG_JOINTS`、`target_overrides=None`（HOME） | 目标关节姿势 |
| `pose_stand_l1` | `pose_l1_penalty` | +5.0 | `joint_indices=_LEG_JOINTS`、`target_overrides=None` | L1 bootstrap：即使远离 HOME 也有常量梯度 |
| `height_stand` | `height_target_gaussian` | +4.0 | `std=0.04`、`target_height=0.138` | 宽高斯 → 从地面拉起 |
| `height_stand_sharp` | `height_target_gaussian` | +4.0 | `std=0.015`、`target_height=0.138` | 窄高斯 → 逼出最后几厘米 |
| `height_stand_l1` | `height_l1_penalty` | +30.0 | `target_height=0.138` | 让「赖在地上」净负（否则就是懒惰最优） |
| `com_upward_velocity` | `com_upward_velocity` | +3.0 | `max_height=0.148` | 奖励站起的*动作*（目标上方 +10 mm 余量，如 `standup` 的 0.125 vs 0.115） |
| `gentle_rise` | `trunk_vertical_accel_penalty` | −0.02 | | 惩罚 `\|a_z\|` → 恒速平滑上升 |
| `upright_linear` | `body_upright_linear` | +6.0 | | `cos(tilt)`：躺着时强梯度 |
| `upright_sharp` | `upright_gaussian_at_height` | +6.0 | `std=0.3`、`height_low=0.075`、`height_high=0.138` | 在高度上 gating 的紧高斯 → 杀掉后倾 |
| `standing_composite` | `standing_composite_score` | +15.0 | `height_std=0.04`、`upright_std=0.40`、`pose_std=0.40`、`target_height=0.138`、`joint_indices=_LEG_JOINTS` | 高度 × 直立 × 姿势 乘性复合分 |

在 `standup` 使用 `asset_cfg=SceneEntityCfg("robot", body_names=("trunk_base",))` 的地方，
所有术语都使用相同配置。

**v1 不加（躯干/头部）冲击惩罚**：`standup` 没有，只有 `velstand` 有。保持最小集。

---

## Observation 与命令

**Observation**：从 roller env 原样继承（61D）。无任何修改 — 这就是从该 env 派生的原因。

在 actor 和 critic 组上加 `nan_policy = "sanitize"`，如同 `roller_slope`：罕见接触
会把 free-joint 发散成 NaN，obs 被消毒（→ 0）以免杀死训练，
出错的 env 在下一步 reset。

**命令**：`twist` 槽被中性化，与 `standup` 完全一致：

```python
command = cfg.commands["twist"]
command.rel_standing_envs = 0.0
command.rel_heading_envs  = 0.0
command.heading_command   = False
command.ranges.heading    = None
command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
command.debug_vis = False
command.ranges.lin_vel_x = (-0.01, 0.01)
command.ranges.lin_vel_y = (-0.01, 0.01)
command.ranges.ang_vel_z = (-0.05, 0.05)
cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
```

`head_pose`（4）和 `body_pose`（6）槽保持 **zero-padded** — roller 家族
（`roller`、`roller_crouch`、`roller_slope`）的约定。这是与行走 `standup`
的一个刻意偏差 — 后者通过真正的 `head_pose` 4D 命令控制头部（见「风险」）。

中性化 twist 的理由：在 `scripts/infer_policy.py` 中，行走 `standup` policy
以 `--standing` 与 `--walking` 一起加载，切换由**速度命令幅值自动触发**
（`infer_policy.py:262`，阈值 0.05）；`standing` 激活时，twist 槽保持为零
（`infer_policy.py:239`）。相位槽（`ground_pick`、`fold`）用于按钮触发的
一次性 trick，不是用来站起。

---

## Reset

添加 `set_ground_state` 事件（`reset` 模式），插入在继承的 `reset_base` 和
`reset_robot_joints` **之后**（事件顺序按 dict 插入顺序）：

```python
cfg.events["set_ground_state"] = EventTermCfg(
    func=microduck_mdp.set_random_ground_state,
    mode="reset",
    params={
        "face_down_prob":  0.50,   # ventre — piloté par le curriculum ci-dessous
        "face_up_prob":    0.00,   # dos — introduit tard (le plus dur)
        "sitting_prob":    0.00,   # pas de bucket assis → aucun override de joint à remapper
        "standing_prob":   0.50,
        "prone_z_min":     0.076,  # cf. note ci-dessous — pas un simple héritage du standup
        "prone_z_max":     0.09,
        "standing_z_min":  0.134,  # roller (contre 0.11–0.12 pour les pieds)
        "standing_z_max":  0.144,
        "sitting_tilt_max": math.radians(10),  # ± bruit de pitch/roll ; s'applique AUSSI au bucket debout
    },
)
```

注意：在 `set_random_ground_state` 中，`standing` 桶复用 `sitting` 桶的四元数 —
所以 `sitting_tilt_max` 也给站立起步加噪，这是有意的。

**关于 `prone_z_min` = 0.076（而非 0.05，后者是从 `standup` 误抄的值）**：趴和仰
共享同一 z 范围，但它们的实测触地高度不同 — 趴 0.0752、仰 0.0475 — 所以单一范围
无法对两者都理想。`standup` 的注释将其下限 `0.05` 归因为**重力稳定后**测得约 0.044
的静止高度；但 reset 时刻重要的是 HOME 姿势下的触地高度，不是倒下后的静止高度。
在 0.05 时，趴着会让躯干外壳**陷地 25 mm**，policy 之后要通过 `gentle_rise` /
`joint_torque_rate_l2` 付出 pushout 代价。`prone_z_min = 0.076` 消除这种互穿，
代价是仰躺以高于静止 28–42 mm 起步 — 这是个比接触 pushout 温和得多的伪影。

**`mdp.py` 无需修改**：基础的 `reset_robot_joints` 用
`joint_names=(".*",)`、`velocity_range=(0.0, 0.0)` 和 `default_joint_vel`（HOME_FRAME
`joint_vel={".*": 0.0}`）→ 4 个被动轮每次 reset 已归零。已验证。

**`ground_state_mix` curriculum**（`event_param_curriculum`），与 `standup` 同样的
easy → hard 逻辑：仰躺晚引入并在末尾获得最多训练。

| iter | 站立 | 趴 | 仰 |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

（步数以 `common_step_counter` 为单位 = `iter × 24`。）

**推力**：`push_robot` 继承自 roller env（±0.2 m/s，间隔 3–6 s）。加上 `standup`
的渐升 curriculum 以免寄生 bootstrap：0 → ±0.08（iter 500）→ ±0.2（iter 1000）。

**终止**：删除 `fell_over`（robot **从摔倒状态起步** — 倾斜终止在此无意义）。
`nan_state` 继承并保留。

**地形**：`plane`。v1 无 rough 变体 — 与 roller env 一致，后者没有 `rough` 参数。

---

## 反向滚动摩擦 curriculum

这是设计中唯一真正新的部分，也是任务所提问题的核心：
**轮子在滚动，纵向没有任何附着力来蹬地。**

机制已存在并继承（`randomize_wheel_friction` 通过 `dr.dof_frictionloss` 作用于
`^passive_.*` + `wheel_friction_curriculum`）。在 roller env 中它**上升** 0 → 0.0015。
这里我们让它**下降**：

| iter | frictionloss | 效果 |
|---|---|---|
| 0 | 0.05 | 轮子几乎锁死 → 像有脚一样站起来 |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | 真实滚动摩擦值（roller env 的值） |

`wheel_friction_curriculum` 只是应用已跨越的最后一级
（`if env.common_step_counter > stage["step"]`）— 下降和上升一样有效。
**零代码要写。**

**这个 curriculum 告诉我们**：如果 `Episode_Reward/standing_composite` 在摩擦下降时
崩溃，就得到清晰答案 — 「有附着力脚」的动作无法迁移到自由轮，需要引导滑冰技术
（中间膝盖支撑、一次一个滑刃）。这是可用的结果，不是失败。

---

## 网络与 PPO

与 `standup` 相同：actor 和 critic `(512, 256, 128)` elu，`obs_normalization=True`
（normalizer 由 `export.py` 烤进 ONNX），PPO `lr=1e-3` adaptive schedule、`desired_kl=0.01`、
`entropy_coef=0.01`、`gamma=0.99`、`lam=0.95`、`num_steps_per_env=24`、`save_interval=250`、
`max_iterations=15_000`。**对称性 OFF**（`SYMMETRY_CFG` 为旧的 51D 布局硬编码，
在 61D 上会崩 — 与所有 v1.5+ env 情况相同）。

---

## 测试

`tests/test_roller_standup_cfg.py`：

1. env 可构建（`play=False` 和 `play=True`）；
2. **`_LEG_JOINTS` / `_NECK_JOINTS` / `_WHEEL_JOINTS` 索引上的关节名正确**
   （针对轮子交错的脆弱性的锁）；
3. 期望的站立 reward 存在，滑冰 reward 缺失
   （`wheel_speed`、`glide`、`single_support`、`feet_flat` 等）；
4. `fell_over` 缺失，`nan_state` 存在；
5. `wheel_friction` curriculum 确为**递减**并终止于 0.0015；
6. `ground_state_mix` curriculum：最后一级概率和为 1 且
   `face_up_prob` 单调递增；
7. **obs 对等**：actor/critic 术语的名称和维度与
   `make_microduck_velocity_rollers_env_cfg()` 相同（否则 ONNX 无法在槽中加载）。

运行：`uv run --with pytest pytest tests/ -q`。

---

## 训练与部署

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
```

监控 `Episode_Reward/standing_composite`（应上升），尤其关注其在**滚动摩擦各级**
（iter 1000/2000/3000/4000）上的表现。

Play：`uv run scripts/play_latest.py`。导出：`uv run scripts/export_latest.py`。

目标部署：policy 在 `--standing` 槽位对 roller policy 在 `--walking`，靠命令幅值
自动切换。**保留**：`infer_policy.py` 是本地 sim/键盘脚本；robot runtime 是
Rust 二进制 `microduck_runtime`，不在本仓库内 — 此处不验证它是否暴露带同样切换的
`--standing` 等价物。交接文档仅列出 `--model`、`--ground-pick`、`--fold-policy`。
待确认。这不影响训练：如果 runtime 无此槽，policy 仍可用在按钮槽
（命令是相位而非零 — 这将是唯一需重新审视的点）。

---

## 风险与关注点

1. **自由轮上站起可能没有专门技术就做不出来。** 这是主要风险。摩擦 curriculum
   设计成清晰地回答这个问题，而不是绕过它。
2. **「仰躺」桶最难。** `standup` 记录该姿势会冻结在「什么都不做」，原因是
   *motion blocker*（`body_ang_vel` 过高、`action_rate` 过强）。这里复用的值
   是「能从任何姿势站起」版本的 — 不要无故加硬。
3. **头部 zero-padded vs `head_pose` 命令。** 若 policy 部署在 `--standing` 且
   有人按头部键，`infer_policy` 写 `cmd[3:7] = head_offset`，policy 看到分布外输入。
   为了保持在 roller 约定内的刻意选择；若站起过程中需要控制头部则需重新审视。
4. **Frictionloss 0.05 远离真实值。** 0 → 2000 iter 各级产生无法迁移的 policy；
   只有最后一级之后（iter 4000+）的 checkpoint 是部署候选。

## 范围外

- 将站起整合到滑行 policy（`velstand` 配方）— 在可行性验证后推迟决定。
- 侧躺起始桶。
- rough / 不平整地形变体。
- 躯干/头部冲击惩罚。
- 对 `roller`、`roller_crouch`、`roller_slope`、`standup`、`velstand` envs 或
  `mdp.py` 的任何修改。
