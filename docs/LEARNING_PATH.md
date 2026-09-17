# MicroDuck RL 学习路径

> 面向第一次接触本仓库、想理解"这个机器人 RL 是怎么训练出来的"的开发者。
> 技术栈：**mjlab (MuJoCo Warp) 并行仿真 + PPO (rsl_rl) + BAM 执行器模型**，61 维 observation 契约，训练后导出 ONNX 部署到真机（sim2real 是本项目的核心目的）。

---

## 第 0 步：先跑起来（半天）

别先读代码，先让环境跑通，建立直觉。

```bash
# 在 microduck_rl 根目录
uv sync                                                  # 装依赖（需 CUDA GPU；ARM 机先 export UV_HTTP_TIMEOUT=600）
uv run list-envs                                         # 列出所有已注册任务

# 关键：先跑 smoke test（64 环境、5 次迭代）——作者说这能抓 95% 的配置错误
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

- 没本地 GPU？任意 train 命令加 `--hf-jobs` 扔到 Hugging Face Jobs 跑云端。
- smoke test 跑通后，再跑一次完整训练（4096 envs，约 1–2 小时出可用步态），同时开 wandb 看曲线。

---

## 第 1 步：啃透一个最小任务 —— Velocity 行走（核心）

所有任务都从 `microduck_velocity_env_cfg.py` 长出来，它就是"主菜谱"。按这个顺序读：

| 看什么 | 文件 | 关注点 |
|---|---|---|
| 任务注册表 | `src/mjlab_microduck/tasks/__init__.py` | 任务 id 怎么注册 |
| **主配置（学习重点）** | `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` | obs / action / reward / DR / 终止条件 |
| 所有 reward/obs/事件函数 | `src/mjlab_microduck/tasks/mdp.py`（约 320KB，最大文件） | 只看 velocity 相关段 |
| 机器人常量 | `src/mjlab_microduck/robot/microduck_constants.py` | 关节布局、HOME 位、BAM 配置 |

读 velocity cfg 时必须搞清楚这 5 件事（任何 RL 控制问题的通用骨架）：

1. **Observation（61 维）**：48 维本体感觉 + 13 维命令块 `[twist(3), head_pose(4), body_pose(6)]`。不用的命令槽位必须零填充，不能删槽位（policy family 热切换契约）。
2. **Action（14 维）**：14 个 Dynamixel XL330 舵机的目标关节角。关节布局：0–4 左腿、5–8 头颈、9–13 右腿。
3. **Reward**：速度跟踪 Gaussian、直立/姿态惩罚、动作平滑惩罚、torque 惩罚……逐项看权重和符号。
4. **Domain Randomization**：电池电压、带载电压跌落、命令延迟、摩擦、IMU 偏差、观测噪声。
5. **终止条件**：倒地、NaN、超时。

---

## 第 2 步：补 PPO 算法本身

本项目用 `rsl_rl` 的现成 PPO，不用从零写算法，但要知道它在干嘛：

- Actor-Critic 结构、GAE 优势估计、clip 截断
- 看配置里的 `RslRl...RunnerCfg`：学习率、clip、entropy、rollout length（每 iteration = 24 × num_envs 步）
- **wandb 读曲线三原则**（作者反复强调）：
  1. mean reward 上升 **且** episode 长度符合任务预期
  2. 每个 `Episode_Reward/<penalty>` 项必须 ≤ 0（否则是符号约定写错了，policy 在"奖励违规"）
  3. 主任务项确实在涨（总 reward 可能光靠正则项涨， trick 根本没学会）

---

## 第 3 步：理解 sim2real 的"真功夫"

这才是本仓库区别于玩具 RL 的地方，按重要性排序：

1. **BAM 执行器模型**（`src/mjlab_microduck/actuator/friction_dr_bam.py`）
   不是理想 PD，而是电压控制 + 反电动势 + Stribeck/Coulomb/带载摩擦。作者原话：800g 双足这个尺度，执行器保真度就是 sim2real 差距的大头。
2. **Observation normalization 烘焙进 ONNX**（`src/mjlab_microduck/export.py`）
   仿真里 `play` 不报错（它内部会套 normalizer），但直接转 checkpoint 到真机就废。**必须走 `scripts/export.py` 导出**。
3. **Backlash 建模**（`src/mjlab_microduck/tasks/backlash.py`、`robot_*_backlash.xml`）
   每个关节串一个被动铰链 `passive_<joint>_backlash` 模拟齿轮间隙，编码器从间隙输出侧读取。
4. **Reward 设计血泪教训** —— 强烈建议逐字读 `AGENTS.md` 第 "Reward design" 一节：
   - 符号约定（自负惩罚函数用正权重，否则奖励违规）
   - reward hacking（RL 会钻一切未明确约束的空子）
   - jackpot 问题（提前到达目标状态后按步发奖 = 鼓励暴力）
   - potential-based shaping（Δ progress 不可被 farming）
   - 正则项分两类：motion-blocker（动态任务要压低）vs smoothness（技能学会后再加）

---

## 第 4 步：动手改一个东西

最有学习效果的练手：

- 改一个 reward 权重（比如把动作平滑权重调大/调小），走 smoke test → 短训练，观察步态变化；
- 或基于 velocity 模板做小改动（比如改 command 采样范围）；
- 改完必须：
  ```bash
  uv run pytest tests/        # CPU 即可跑，锁死 cfg 不变量
  uv run train <TASK> --env.scene.num-envs 64 --agent.max_iterations 5   # smoke test
  ```

---

## 常用命令速查

```bash
uv run list-envs                                                          # 看任务注册表
uv run train <TASK_ID> --env.scene.num-envs 4096                           # 正式训练
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5    # smoke test（必跑）
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>           # 仿真里看训练结果
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>                 # 导出 ONNX（烘焙 normalizer）
uv run scripts/infer_policy.py --walking out.onnx                         # CPU MuJoCo 里模拟真机部署
uv run --with pytest pytest/                                              # 回归测试（CPU）

# 断点续训
uv run train <TASK_ID> --env.scene.num-envs 4096 \
    --agent.load-checkpoint model_XXXX.pt --agent.resume True
```

---

## 一句话路线

> **smoke test 跑通 → 死磕 `microduck_velocity_env_cfg.py` + `mdp.py` 的 velocity 段 → 读 `AGENTS.md` 的 Reward design 和 Sim2real footguns 两节 → 改个 reward 权重看步态变化。**

`AGENTS.md` 是作者写的"distilled playbook"，所有约定和坑都有，建议作为枕边书反复翻。
