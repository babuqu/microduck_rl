# 通过姿态跟踪实现的 shoot 任务 — 实现计划

> **对于 agentic workers：** 必需的 SUB-SKILL：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现本计划。步骤使用 checkbox (`- [ ]`) 语法进行跟踪。

**目标：** 添加一个 RL 任务 `Mjlab-Shoot-Flat-MicroDuck`，通过相位插值的 4 个 keyframe 姿态轨迹（STAND → PIED_ARRIÈRE → PIED_AVANT → STAND）来学习一个 one-shot shoot 手势（右腿）。

**架构：** 与本分支的 `ground_pick` 任务相同的模板。一个相位命令 (`GroundPickPhaseCommand`，`[cos,sin,0]`) 驱动在 3 个姿态之间插值的关节目标；高斯 + L1 rewards 奖励跟踪；61D 统一 obs 用于在 runtime 的按钮 slot 中部署。无模拟球。

**技术栈：** Python、PyTorch、mjlab 1.3.0、MuJoCo、uv、pytest。

## 全局约束

- Obs **61D 统一**，与其他 microduck policies 相同 (`[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`，head+body command zero-padding)。不要破坏此形状。
- 关节解析**按名称** (`asset.find_joints([name])`)，绝不通过硬编码索引。
- **14 个**活动关节（mouth 除外）。机器人 `MICRODUCK_WALK_ROBOT_CFG`。
- 不要以破坏性方式修改 Rust runtime 或命令类：添加的 `randomize_phase` 标志必须默认为 `True` 以保持 `ground_pick` 兼容。
- **右**腿踢击，**左**腿支撑。
- 测试：`uv run --with pytest pytest tests/ -q`。
- Commit 约定：法语消息，风格为 `feat:`/`docs:`/`test:`。

---

## 文件结构

- `src/mjlab_microduck/tasks/mdp.py` — 修改：添加 `kick_pose_target`（纯函数）、`_kick_pose_error`、`kick_pose_track`、`kick_pose_track_l1`；向 `GroundPickPhaseCommand` / `GroundPickPhaseCommandCfg` 添加 `randomize_phase` 标志。
- `src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py` — 创建：`make_microduck_shoot_env_cfg`、`MicroduckShootRlCfg`、`STAND_POSE`/`KICK_BACK_POSE`/`KICK_FWD_POSE`、timings。
- `src/mjlab_microduck/tasks/__init__.py` — 修改：import + `register_mjlab_task("Mjlab-Shoot-Flat-MicroDuck", …)`。
- `tests/test_shoot.py` — 创建：纯函数测试 (`kick_pose_target`) + 通过 stub-env 的 rewards 测试。
- `tests/test_shoot_cfg.py` — 创建：集成测试（env 构建成功、正确的命令/rewards）。

---

### 任务 1：在相位命令上添加 `randomize_phase` 标志

**文件：**
- 修改：`src/mjlab_microduck/tasks/mdp.py:3618-3672` (`GroundPickPhaseCommand` + `GroundPickPhaseCommandCfg`)
- 测试：`tests/test_shoot.py`

**接口：**
- 产出：`GroundPickPhaseCommandCfg(randomize_phase: bool = True, period: float = 4.0, …)`；运行时 `reset()` 在 `randomize_phase=False` 时将 φ=0，否则为 `rand()`。

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_shoot.py`，内容如下：

```python
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg


def test_phase_cmd_randomize_flag_default_true():
    cfg = GroundPickPhaseCommandCfg()
    assert cfg.randomize_phase is True


def test_phase_cmd_randomize_flag_settable_false():
    cfg = GroundPickPhaseCommandCfg(randomize_phase=False)
    assert cfg.randomize_phase is False
```

- [ ] **步骤 2：运行测试，验证失败**

运行：`uv run --with pytest pytest tests/test_shoot.py -q`
预期：FAIL — `TypeError: __init__() got an unexpected keyword argument 'randomize_phase'`。

- [ ] **步骤 3：向 cfg 添加字段并在类中串联**

在 `GroundPickPhaseCommandCfg`（dataclass，约第 3667 行）添加字段：

```python
@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # cycle length in seconds; sitstand uses 8.0
    randomize_phase: bool = True  # False -> chaque épisode démarre à φ=0 (STAND)

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)
```

在 `GroundPickPhaseCommand.__init__`（约第 3634 行）读取标志：

```python
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)
        self._period = float(getattr(cfg, "period", self.PERIOD))
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))
```

在 `GroundPickPhaseCommand.reset`（约第 3649 行）遵守标志：

```python
    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            if self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}
```

- [ ] **步骤 4：运行测试，验证成功**

运行：`uv run --with pytest pytest tests/test_shoot.py -q`
预期：PASS（2 个测试）。

- [ ] **步骤 5：Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_shoot.py
git commit -m "feat: flag randomize_phase sur GroundPickPhaseCommand (défaut True)"
```

---

### 任务 2：纯函数 `kick_pose_target`

**文件：**
- 修改：`src/mjlab_microduck/tasks/mdp.py`（在 `phase_pose_blend` 附近添加，约第 2062 行）
- 测试：`tests/test_shoot.py`

**接口：**
- 产出：`kick_pose_target(phase: Tensor(B,), stand, back, forward, windup_end: float, kick_end: float, return_end: float) -> Tensor(B,k)`。`stand/back/forward` 是 `(k,)` 或 `(1,k)` 张量。段：[0,windup_end) STAND→BACK, [windup_end,kick_end) BACK→FORWARD, [kick_end,return_end) FORWARD→STAND, [return_end,1) STAND。

- [ ] **步骤 1：编写失败的测试**

添加到 `tests/test_shoot.py`：

```python
import torch
from mjlab_microduck.tasks.mdp import kick_pose_target

W, K, R = 0.35, 0.45, 0.75  # windup_end, kick_end, return_end
STAND = torch.tensor([0.0, 0.0])
BACK = torch.tensor([1.0, -1.0])
FWD = torch.tensor([-1.0, 2.0])


def _t(phase):
    return kick_pose_target(torch.tensor([phase]), STAND, BACK, FWD, W, K, R)[0]


def test_kick_target_keypoints():
    assert torch.allclose(_t(0.0), STAND)          # début: STAND
    assert torch.allclose(_t(W), BACK)             # fin armement: BACK
    assert torch.allclose(_t(K), FWD)              # fin frappe: FORWARD
    assert torch.allclose(_t(R), STAND)            # fin retour: STAND
    assert torch.allclose(_t(0.9), STAND)          # repos: STAND


def test_kick_target_midsegments():
    assert torch.allclose(_t(W / 2), 0.5 * BACK)                    # mi-armement
    assert torch.allclose(_t((W + K) / 2), 0.5 * (BACK + FWD))      # mi-frappe
    assert torch.allclose(_t((K + R) / 2), 0.5 * FWD)              # mi-retour


def test_kick_target_batch_shape():
    phase = torch.linspace(0.0, 1.0, 50)
    out = kick_pose_target(phase, STAND, BACK, FWD, W, K, R)
    assert out.shape == (50, 2)
    # chaque composante reste dans l'enveloppe des 3 poses
    lo = torch.minimum(torch.minimum(STAND, BACK), FWD)
    hi = torch.maximum(torch.maximum(STAND, BACK), FWD)
    assert (out >= lo - 1e-6).all() and (out <= hi + 1e-6).all()
```

- [ ] **步骤 2：运行，验证失败**

运行：`uv run --with pytest pytest tests/test_shoot.py -q`
预期：FAIL — `ImportError: cannot import name 'kick_pose_target'`。

- [ ] **步骤 3：实现纯函数**

在 `mdp.py` 中 `phase_pose_blend`（约第 2062 行）之后添加：

```python
def kick_pose_target(
    phase: torch.Tensor,
    stand: torch.Tensor,
    back: torch.Tensor,
    forward: torch.Tensor,
    windup_end: float,
    kick_end: float,
    return_end: float,
) -> torch.Tensor:
    """Cible articulaire interpolée d'un geste de shoot à 4 keyframes.

    phase (B,) ∈ [0,1). stand/back/forward (k,) ou (1,k). Retour (B,k).

    [0, windup_end)        STAND   -> BACK     (armement)
    [windup_end, kick_end) BACK    -> FORWARD  (frappe sèche)
    [kick_end, return_end) FORWARD -> STAND    (retour)
    [return_end, 1.0)      STAND             (repos)
    """
    p = phase.unsqueeze(-1)  # (B,1)

    def interp(a, b, s):
        return a + s * (b - a)

    s1 = (p / windup_end).clamp(0.0, 1.0)
    s2 = ((p - windup_end) / (kick_end - windup_end)).clamp(0.0, 1.0)
    s3 = ((p - kick_end) / (return_end - kick_end)).clamp(0.0, 1.0)

    seg1 = interp(stand, back, s1)
    seg2 = interp(back, forward, s2)
    seg3 = interp(forward, stand, s3)  # à s3=1 (phase>=return_end) => STAND

    out = seg1
    out = torch.where(p >= windup_end, seg2, out)
    out = torch.where(p >= kick_end, seg3, out)
    return out
```

- [ ] **步骤 4：运行，验证成功**

运行：`uv run --with pytest pytest tests/test_shoot.py -q`
预期：PASS（所有 kick_target 测试）。

- [ ] **步骤 5：Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_shoot.py
git commit -m "feat: kick_pose_target — cible interpolée du geste de shoot (4 keyframes)"
```

---

### 任务 3：跟踪 rewards `kick_pose_track` / `kick_pose_track_l1`

**文件：**
- 修改：`src/mjlab_microduck/tasks/mdp.py`（在 `kick_pose_target` 之后添加）
- 测试：`tests/test_shoot.py`

**接口：**
- 消费：`kick_pose_target`（任务 2）。
- 产出：
  - `kick_pose_track(env, command_name="twist", stand_pose=None, back_pose=None, forward_pose=None, std=0.4, windup_end=0.35, kick_end=0.45, return_end=0.75, asset_cfg=_DEFAULT_ASSET_CFG) -> Tensor(B,)` — 高斯 `exp(-((q-cible)/std)²).mean`。
  - `kick_pose_track_l1(env, …mêmes args sauf std) -> Tensor(B,)` — `-(|q-cible|).mean`。
  - Helper `_kick_pose_error(env, asset_cfg, command_name, stand_pose, back_pose, forward_pose, windup_end, kick_end, return_end) -> (cur, target)`。

- [ ] **步骤 1：编写失败的测试（stub-env）**

添加到 `tests/test_shoot.py`：

```python
from mjlab_microduck.tasks.mdp import kick_pose_track, kick_pose_track_l1

STAND_D = {"a": 0.0, "b": 0.0}
BACK_D = {"a": 1.0, "b": -1.0}
FWD_D = {"a": -1.0, "b": 2.0}
_IDX = {"a": 0, "b": 1}


class _FakeData:
    def __init__(self, joint_pos):
        self.joint_pos = joint_pos
        self.default_joint_pos = torch.zeros_like(joint_pos)


class _FakeAsset:
    def __init__(self, joint_pos):
        self.data = _FakeData(joint_pos)

    def find_joints(self, names):
        return ([_IDX[names[0]]], names)


class _FakeScene:
    def __init__(self, asset):
        self._a = asset

    def __getitem__(self, name):
        return self._a


class _FakeCmdMgr:
    def __init__(self, cmd):
        self._cmd = cmd

    def get_command(self, name):
        return self._cmd


class _FakeEnv:
    def __init__(self, joint_pos, phase):
        self.scene = _FakeScene(_FakeAsset(joint_pos))
        # cmd = [cos, sin, 0]
        cmd = torch.stack(
            [torch.cos(2 * torch.pi * phase), torch.sin(2 * torch.pi * phase),
             torch.zeros_like(phase)], dim=-1)
        self.command_manager = _FakeCmdMgr(cmd)
        self.device = "cpu"
        self.num_envs = joint_pos.shape[0]


def test_kick_track_perfect_at_stand_phase():
    # phase=0 -> cible STAND=[0,0] ; joint_pos exactement STAND -> reward ~1
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
    r = kick_pose_track(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-4)


def test_kick_track_lower_when_off_target():
    # phase=0.45 (kick_end) -> cible FORWARD=[-1,2] ; joint_pos=STAND -> reward < 0.5
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.45]))
    r = kick_pose_track(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert (r < 0.5).all()


def test_kick_track_l1_zero_when_perfect():
    env = _FakeEnv(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
    r = kick_pose_track_l1(env, stand_pose=STAND_D, back_pose=BACK_D, forward_pose=FWD_D)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6)
```

- [ ] **步骤 2：运行，验证失败**

运行：`uv run --with pytest pytest tests/test_shoot.py -q`
预期：FAIL — `ImportError: cannot import name 'kick_pose_track'`。

- [ ] **步骤 3：实现 helper + rewards**

在 `mdp.py` 中 `kick_pose_target` 之后添加：

```python
def _kick_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_pose: dict,
    back_pose: dict,
    forward_pose: dict,
    windup_end: float,
    kick_end: float,
    return_end: float,
):
    """(cur, target) pour le geste de shoot, joints résolus PAR NOM.

    Les 3 poses partagent les mêmes clés (14 joints). L'ordre des noms est
    donné par `stand_pose`.
    """
    if not stand_pose:
        raise ValueError("_kick_pose_error requires a non-empty stand_pose dict")
    asset: Entity = env.scene[asset_cfg.name]
    names = list(stand_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]

    def vec(d):
        return torch.tensor([d[n] for n in names], device=env.device,
                            dtype=asset.data.joint_pos.dtype)

    stand_v, back_v, fwd_v = vec(stand_pose), vec(back_pose), vec(forward_pose)

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    target = kick_pose_target(phase, stand_v, back_v, fwd_v,
                              windup_end, kick_end, return_end)          # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def kick_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    std: float = 0.4,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussienne sur la pose articulaire vs cible interpolée du shoot.

    Reward directif et symétrique : chaque phase impose la config articulaire
    exacte. Résolution PAR NOM.
    """
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def kick_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1 vers la cible interpolée (gradient constant, pénalité<=0)."""
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end,
    )
    return -(cur - target).abs().mean(dim=-1)
```

- [ ] **步骤 4：运行，验证成功**

运行：`uv run --with pytest pytest tests/test_shoot.py -q`
预期：PASS（所有测试，包括新增的 3 个）。

- [ ] **步骤 5：Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_shoot.py
git commit -m "feat: rewards kick_pose_track + kick_pose_track_l1 (suivi du geste de shoot)"
```

---

### 任务 4：Env config `microduck_shoot_env_cfg.py`

**文件：**
- 创建：`src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py`
- 测试：（通过任务 5）

**接口：**
- 消费：`kick_pose_track`、`kick_pose_track_l1`（任务 3）；`GroundPickPhaseCommandCfg(randomize_phase=…)`（任务 1）；`feet_grounded_reward`、`feet_flat_penalty`、`neck_action_rate_l2`、`joint_torques_l2`、`zero_command_padding`、`robot_state_is_nan`、DR events（已存在于 `mdp.py`）。
- 产出：`make_microduck_shoot_env_cfg(play=False, rough=False) -> ManagerBasedRlEnvCfg`；`MicroduckShootRlCfg`；常量 `SHOOT_PERIOD`、`WINDUP_END`、`KICK_END`、`RETURN_END`、`STAND_POSE`、`KICK_BACK_POSE`、`KICK_FWD_POSE`。

- [ ] **步骤 1：以 ground_pick 文件为基础**

```bash
cp src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py \
   src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py
```

此文件已经提供所有需要原样保留的 sim2real boilerplate：DR（CoM、head CoM、mass/inertia、friction BAM、armature、IMU misalignment obs-level、encoder-bias、pushes）、61D obs 块（actor 的 `del base_lin_vel`、critic 的 base_lin_vel、移除 `foot_height`/`height_scan`、delays/noise、`head_command`/`body_command` zero-padding）、`nan_state` 终止条件、`expand_bam_friction_fields` / `reset_action_history` events、action_rate/CoM curriculum。只修改：robot cfg、sensors、command 和 rewards 块。

- [ ] **步骤 2：调整文件头、函数名和常量**

将顶部 docstring 替换为 shoot 描述，并在 `def make_microduck_ground_pick_env_cfg` 之前添加常量 + poses（占位符 — 用 `read_pose.py` 读取结果替换）。将函数重命名为 `make_microduck_shoot_env_cfg`。

```python
# ── Timings du geste (phase normalisée [0,1)) ────────────────────────────────
SHOOT_PERIOD = 2.5   # s — durée d'un cycle (doit matcher --ground-pick-period au déploiement)
WINDUP_END = 0.35    # STAND -> BACK
KICK_END = 0.45      # BACK -> FORWARD (segment court = frappe sèche)
RETURN_END = 0.75    # FORWARD -> STAND, puis repos jusqu'à 1.0

# ── Poses (rad, 14 joints, mouth exclu) ──────────────────────────────────────
# Convention: jambe droite frappe (hanche/genou droit actifs), gauche en appui.
# STAND_POSE = pose HOME du sim (HOME_FRAME / default_joint_pos) pour que φ=0
# coïncide avec la config de reset (invariant randomize_phase=False). BACK/FWD
# sont des PLACEHOLDERS jambe droite, à affiner via read_pose.py.
STAND_POSE = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": -0.0049, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.4530,
}
KICK_BACK_POSE = {  # armement: hanche droite en extension arrière + genou fléchi
    **STAND_POSE,
    "right_hip_pitch": -0.6,
    "right_knee": 0.8,
    "right_ankle": -0.2,
}
KICK_FWD_POSE = {  # frappe: hanche droite fléchie avant + genou tendu
    **STAND_POSE,
    "right_hip_pitch": 0.7,
    "right_knee": -0.1,
    "right_ankle": 0.1,
}
```

> NOTE 给 pose 采集者：用 `read_pose.py` 读取结果替换这些值（断电、手动将机器人放置在每个位置）。在 3 个 dict 中保持 14 个键相同。

- [ ] **步骤 3：Robot cfg 和 import**

在 imports 中，将 `MICRODUCK_GROUND_PICK_ROBOT_CFG` 替换为 `MICRODUCK_WALK_ROBOT_CFG`：

```python
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
```

在函数中，entities 行：

```python
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
```

- [ ] **步骤 4：Sensors — 保留 self_collision，替换脚部 sensors**

将 `feet_ground_contact` sensor（双脚）定义替换为**仅左脚** sensor（支撑脚），并删除 `head_impact_cfg` sensor（此处无用）。`self_collision_cfg` sensor 保留。

```python
    left_foot_ground_cfg = ContactSensorCfg(
        name="left_foot_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^left_foot_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
```

以及 scene sensors 行：

```python
    cfg.scene.sensors = (left_foot_ground_cfg, self_collision_cfg)
```

删除 `head_impact_cfg` 的定义和所有引用（`head_impact_penalty` reward 将在步骤 6 移除）。

- [ ] **步骤 5：相位命令（randomize_phase=False，shoot 周期）**

将命令块（创建 `GroundPickPhaseCommandCfg` 的那个）替换为：

```python
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand}
    )
    cfg.commands["twist"].period = SHOOT_PERIOD
    cfg.commands["twist"].randomize_phase = False
```

- [ ] **步骤 6：Rewards — 移除 ground_pick，添加 shoot**

删除 ground_pick 专属 rewards：`mouth_ground_proximity`、`mouth_perpendicular_to_ground`、`ground_pick_return_pose_legs`、`ground_pick_return_pose_neck`、`feet_grounded`（双脚）、`head_impact_penalty`。替换为 shoot 块：

```python
    # ── Objectif : suivi de la pose interpolée du shoot ───────────────────────
    _pose_params = {
        "command_name": "twist",
        "stand_pose": STAND_POSE,
        "back_pose": KICK_BACK_POSE,
        "forward_pose": KICK_FWD_POSE,
        "windup_end": WINDUP_END,
        "kick_end": KICK_END,
        "return_end": RETURN_END,
    }
    cfg.rewards["kick_pose_track"] = RewardTermCfg(
        func=microduck_mdp.kick_pose_track,
        weight=6.0,
        params={**_pose_params, "std": 0.4},
    )
    cfg.rewards["kick_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.kick_pose_track_l1,
        weight=2.0,
        params=dict(_pose_params),
    )

    # ── Équilibre / appui (jambe unique) ──────────────────────────────────────
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    # Pied GAUCHE planté (appui). feet_grounded_reward avec un capteur mono-pied
    # -> found ∈ {0,1} -> reward ∈ {0,0.5} ; poids 6.0 => contribution max ~3.0.
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=6.0,
        params={"sensor_name": left_foot_ground_cfg.name},
    )

    # Pied gauche à plat.
    cfg.rewards["feet_flat_left"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot",))},
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
```

- [ ] **步骤 7：轻量化 regularization（允许 snap 通过）**

ground_pick 文件设置 `action_rate_l2=-2.0`、`neck_action_rate_l2=-1.0`、`joint_torques_l2=-5e-3` + 一个最终达到 -2.0 的 action_rate curriculum。对于 shoot 我们轻量化。将这 3 个块替换为：

```python
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-0.5
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )
```

并轻量化 action_rate curriculum（保留结构，目标 -0.5）：

```python
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,        "weight": -0.2},
                {"step": 250 * 24, "weight": -0.4},
                {"step": 500 * 24, "weight": -0.5},
            ],
        },
    )
```

- [ ] **步骤 8：Reset — 站立高度**

保留**站立高度** `(0.12, 0.13)` — 这是 velocity env（步行）和 ground_pick 的值。⚠️ 这不是加性偏移「蹲姿」：`InitialStateCfg` 的默认根 `pos` 是 (0,0,0)，所以 reset 高度是 z ∈ [0.12, 0.13] m **绝对值** = 站立（无跌落）。检查/设置为：

```python
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)
```

（不要注入入口速度 — 这是站立 shoot，不是滑动。）

- [ ] **步骤 9：重命名 RlCfg**

在文件底部，将 `MicroduckGroundPickRlCfg` 重命名为 `MicroduckShootRlCfg` 并更改 experiment 名称：

```python
MicroduckShootRlCfg = RslRlOnPolicyRunnerCfg(
    # … (garder actor/critic/algorithm identiques) …
    wandb_project="mjlab_microduck",
    experiment_name="shoot",
    run_name="shoot",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
```

- [ ] **步骤 10：验证模块可导入**

运行：`uv run python -c "from mjlab_microduck.tasks.microduck_shoot_env_cfg import make_microduck_shoot_env_cfg, MicroduckShootRlCfg; print('ok')"`
预期：`ok`（无 ImportError / NameError — 特别是不再有任何对 `head_impact_cfg`、`MICRODUCK_GROUND_PICK_ROBOT_CFG` 或已删除 ground_pick rewards 的引用）。

- [ ] **步骤 11：Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py
git commit -m "feat: env config Mjlab-Shoot (geste de shoot par suivi de poses)"
```

---

### 任务 5：注册 + 集成测试

**文件：**
- 修改：`src/mjlab_microduck/tasks/__init__.py`
- 测试：`tests/test_shoot_cfg.py`

**接口：**
- 消费：`make_microduck_shoot_env_cfg`、`MicroduckShootRlCfg`（任务 4）。
- 产出：注册的任务 `Mjlab-Shoot-Flat-MicroDuck`。

- [ ] **步骤 1：编写失败的集成测试**

创建 `tests/test_shoot_cfg.py`：

```python
from mjlab_microduck.tasks.microduck_shoot_env_cfg import (
    make_microduck_shoot_env_cfg,
    STAND_POSE, KICK_BACK_POSE, KICK_FWD_POSE, SHOOT_PERIOD,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_poses_have_same_14_keys():
    assert set(STAND_POSE) == set(KICK_BACK_POSE) == set(KICK_FWD_POSE)
    assert len(STAND_POSE) == 14
    assert "mouth" not in STAND_POSE


def test_shoot_cfg_builds_with_phase_command():
    cfg = make_microduck_shoot_env_cfg()
    twist = cfg.commands["twist"]
    assert isinstance(twist, microduck_mdp.GroundPickPhaseCommandCfg)
    assert twist.randomize_phase is False
    assert twist.period == SHOOT_PERIOD


def test_shoot_cfg_has_kick_rewards_and_no_walking():
    cfg = make_microduck_shoot_env_cfg()
    assert "kick_pose_track" in cfg.rewards
    assert "kick_pose_l1" in cfg.rewards
    assert "support_foot_grounded" in cfg.rewards
    for gone in ("track_linear_velocity", "track_angular_velocity",
                 "mouth_ground_proximity", "ground_pick_return_pose_legs"):
        assert gone not in cfg.rewards
```

- [ ] **步骤 2：运行，验证失败**

运行：`uv run --with pytest pytest tests/test_shoot_cfg.py -q`
预期：poses 测试可能 PASS，但整体只有在 env 无错误构建后才变绿；如果 `make_...` 抛异常，则 FAIL。（此时通过任务 4 文件 import 已可工作。）

- [ ] **步骤 3：注册任务**

在 `src/mjlab_microduck/tasks/__init__.py` 中 ground_pick import 块之后（约第 50 行）添加：

```python
from .microduck_shoot_env_cfg import (
    make_microduck_shoot_env_cfg,
    MicroduckShootRlCfg,
)
```

在 GroundPick-Rough 的 `register_mjlab_task` 块之后（约第 161 行）添加：

```python
register_mjlab_task(
    task_id="Mjlab-Shoot-Flat-MicroDuck",
    env_cfg=make_microduck_shoot_env_cfg(),
    play_env_cfg=make_microduck_shoot_env_cfg(play=True),
    rl_cfg=MicroduckShootRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Shoot task registered: Mjlab-Shoot-Flat-MicroDuck")
```

- [ ] **步骤 4：运行全部，验证成功**

运行：`uv run --with pytest pytest tests/ -q`
预期：PASS（test_shoot.py + test_shoot_cfg.py + 现有测试）。

- [ ] **步骤 5：验证任务注册**

运行：`uv run python -c "import mjlab_microduck.tasks"`
预期：输出包含 `✓ Shoot task registered: Mjlab-Shoot-Flat-MicroDuck`。

- [ ] **步骤 6：Commit**

```bash
git add src/mjlab_microduck/tasks/__init__.py tests/test_shoot_cfg.py
git commit -m "feat: enregistre Mjlab-Shoot-Flat-MicroDuck + test d'intégration"
```

---

## 实现之后（TDD 计划之外）

1. **用 `read_pose.py` 采集真实 poses**（STAND、PIED_ARRIÈRE、PIED_AVANT），替换 `microduck_shoot_env_cfg.py` 中的占位符。
2. **训练**：`uv run train Mjlab-Shoot-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 8000`。监控 `Episode_Reward/kick_pose_track`（应上升）。
3. **Play**：play_latest 脚本；验证踢击时左脚的平衡。
4. **Export ONNX** + 部署到 phase slot（`--ground-pick shoot.onnx --ground-pick-period 2.5 --ground-pick-kp-ratio 1.0`）。
5. **可能的调整**：周期/timings（snap）、`action_rate` 权重，以及如果跟踪缺乏力度时可能添加的「脚向前速度」reward（踢击段）。

## 自审 — spec 覆盖

- 文件 & 注册 → 任务 4、5。✅
- 14 关节占位 poses → 任务 4 步骤 2，任务 5 测试。✅
- 相位命令 + `randomize_phase=False` + 周期 → 任务 1、任务 4 步骤 5，任务 5 测试。✅
- `kick_pose_target` + `kick_pose_track` + `kick_pose_track_l1` → 任务 2、3。✅
- 平衡/支撑（upright、左脚踩实、左脚 feet_flat、self_collisions、body_ang_vel）→ 任务 4 步骤 6。✅
- 轻量化 regularization → 任务 4 步骤 7。✅
- 61D obs 对齐（继承自 ground_pick，保留）→ 任务 4 步骤 1。✅
- 纯函数 + cfg 测试 → 任务 2、3、5。✅
