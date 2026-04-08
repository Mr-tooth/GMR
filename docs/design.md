# Motion Dataset Player / Editor — 设计文档 (Design Doc)

> 版本：v0.1  
> 状态：草稿  
> 最后更新：2025-04

---

## 1. 架构总览

```
motion-player
├── motion_player/
│   ├── core/
│   │   ├── models.py        # 数据模型（StandardMotion dataclass）
│   │   ├── adapters.py      # DatasetAdapter / ModelAdapter / DOF 审查
│   │   ├── playback.py      # PlaybackEngine（帧调度、状态机）
│   │   ├── editing.py       # EditingEngine（帧编辑、区间编辑、传播）
│   │   └── metrics.py       # MetricsEngine（可插拔 MetricTerm）
│   ├── backends/
│   │   ├── mujoco_backend.py  # MuJoCo kinematic replay + viewer
│   │   └── nv_backend.py      # NVIDIA HumanoidViewMotion（可选）
│   ├── cli.py               # 命令行入口
│   └── __init__.py
├── pyproject.toml
└── README_motion_player.md
```

**核心设计原则：**
- `core/` 与任何渲染后端**完全解耦**，所有计算逻辑不 import `mujoco` 或 Isaac
- 后端（`backends/`）实现统一的 `BaseRenderer` 接口
- 数据流：`StandardMotion`（内存对象）← `DatasetAdapter`（IO + 校验）← 磁盘文件
- 编辑操作（`EditOperation`）为可序列化 dataclass，支持撤销栈和脚本化

---

## 2. 数据模型

### 2.1 `StandardMotion`

```python
@dataclass
class StandardMotion:
    """内存中的标准动作表示。

    所有数组均为 numpy float32，四元数约定为 xyzw（scalar-last）。
    N_eff = motion_length = 原始 N-1 帧（AMP N-1 对齐语义）。
    """
    fps: float
    motion_length: int               # N_eff = N - 1
    motion_weight: float

    root_pos: np.ndarray             # (N_eff, 3)
    root_rot: np.ndarray             # (N_eff, 4)  xyzw
    projected_gravity: np.ndarray    # (N_eff, 3)
    root_lin_vel: np.ndarray         # (N_eff, 3)
    root_ang_vel: np.ndarray         # (N_eff, 3)
    dof_pos: np.ndarray              # (N_eff, num_dofs)
    dof_vel: np.ndarray              # (N_eff, num_dofs)
    key_body_pos_local: np.ndarray   # (N_eff, K*3)  全部 body

    # 可选元数据（由 sidecar YAML 填充）
    dof_names: list[str] | None = None          # 长度 = num_dofs
    key_body_names: list[str] | None = None     # 长度 = K
    robot: str | None = None
    source_pipeline: str | None = None
    gmr_ik_config: str | None = None
```

### 2.2 Sidecar YAML 格式

```yaml
# <clip_name>_meta.yaml
dof_names: [left_hip_yaw, left_hip_roll, ..., right_ankle_pitch]
key_body_names: [pelvis, left_thigh, ..., right_foot, left_hand, right_hand]
robot: booster_t1
source_pipeline: gmr_bvh_lafan1
gmr_ik_config: bvh_lafan1_to_t1_29dof
```

**设计决策：** sidecar 与主数据文件分离，保持标准文件向后兼容（不修改 `rsl-rl-ex` 现有输出格式）。

### 2.3 `EditState`

```python
@dataclass
class EditState:
    """单个编辑操作的快照，用于撤销栈。"""
    frame_idx: int
    field: str        # 'root_pos' | 'root_rot' | 'dof_pos'
    before: np.ndarray
    after: np.ndarray
```

---

## 3. 模块设计

### 3.1 DatasetAdapter

**职责：** 从磁盘加载标准文件，校验 schema，填充 `StandardMotion` 对象。

```python
class DatasetAdapter:
    def load(self, path: str | Path,
             sidecar_path: str | Path | None = None) -> StandardMotion:
        """加载单个 clip，自动检测 .pkl / .npy 格式。
        
        - 校验必须字段存在
        - quat 归一化（root_rot）
        - 若提供 sidecar，填充 dof_names / key_body_names 等元数据
        """

    def load_dataset(self, directory: str | Path) -> list[StandardMotion]:
        """批量加载目录下所有标准文件。"""

    def save(self, motion: StandardMotion, path: str | Path) -> None:
        """以 .pkl 格式保存 StandardMotion，保持与原始 schema 兼容。"""
```

### 3.2 ModelAdapter

**职责：** 加载机器人模型（MJCF/URDF），建立 DOF 映射关系。

```python
class ModelAdapter:
    def load_mjcf(self, xml_path: str | Path,
                  mapping_config: dict | None = None) -> RobotModel:
        """加载 MJCF，建立 joint_name -> qpos_index 映射。
        
        mapping_config 来自 mapping.yaml，支持：
        - dof_order_in_dataset: list[str]（数据集 DOF 顺序）
        - name_map: dict[str, str]（数据集名 -> MJCF 名）
        - sign_flip: dict[str, float]（符号翻转 +1/-1）
        - offset: dict[str, float]（零位偏置，单位 rad）
        """

    def motion_to_qpos(self, motion: StandardMotion,
                       frame_idx: int) -> np.ndarray:
        """将 StandardMotion 的第 frame_idx 帧转为 MuJoCo qpos。
        
        MuJoCo free joint qpos layout: [x, y, z, w, qx, qy, qz, dof_0, ...]
        注意 MuJoCo 内部四元数为 wxyz（scalar-first），需要从 xyzw 转换。
        """
```

### 3.3 DOF 顺序审查与修复模块

**职责：** 检测数据集 DOF 顺序与机器人模型 DOF 顺序的不一致，生成 sidecar YAML。

```python
class DOFAuditor:
    def audit(self, motion: StandardMotion,
              robot_model: RobotModel) -> DOFAuditReport:
        """对比数据集 DOF 与 MJCF joint 顺序，输出差异报告。"""

    def generate_sidecar(self, source_pkl_path: Path,
                         robot_xml_path: Path) -> dict:
        """从 GMR 输出元数据推断并生成 sidecar YAML 内容。"""

    def repair(self, motion: StandardMotion,
               canonical_order: list[str]) -> StandardMotion:
        """按 canonical_order 重排 dof_pos / dof_vel 列。"""
```

**DOFAuditReport 字段：**
- `matched: list[str]` — 名称和顺序均匹配的 DOF
- `mismatched: list[tuple[str, str]]` — (数据集名, 模型名) 顺序不同
- `unmatched_in_data: list[str]` — 数据集有但模型无
- `unmatched_in_model: list[str]` — 模型有但数据集无
- `is_order_compatible: bool`

### 3.4 PlaybackEngine

**职责：** 管理播放状态机，调度帧更新。

```python
class PlaybackEngine:
    def __init__(self, motion: StandardMotion, fps_override: float | None = None):
        """初始化。fps_override 为 None 时使用 motion.fps。"""

    # 状态控制
    def play(self) -> None
    def pause(self) -> None
    def reset(self) -> None
    def step(self, delta: int = 1) -> None     # 正数=前进，负数=后退
    def seek(self, frame_idx: int) -> None

    # 属性
    @property
    def current_frame(self) -> int
    @property
    def is_playing(self) -> bool
    @property
    def speed(self) -> float                   # 播放速度倍率

    # 回调接口（供后端注册）
    def on_frame_change(self, callback: Callable[[int, StandardMotion], None]):
        """注册帧变更回调。backend 通过此接口接收帧更新通知。"""

    def tick(self) -> bool:
        """推进播放状态一步（由主循环调用）。
        返回 True 若帧发生了变化。
        """
```

### 3.5 EditingEngine

**职责：** 管理编辑操作、撤销栈与跨帧传播。

```python
class EditingEngine:
    def __init__(self, motion: StandardMotion):
        self._motion = motion
        self._undo_stack: list[EditState] = []
        self._redo_stack: list[EditState] = []

    # 单帧编辑
    def edit_root_pos(self, frame_idx: int, delta: np.ndarray) -> None
    def edit_root_rot_euler(self, frame_idx: int,
                            delta_rpy: np.ndarray) -> None
    def edit_dof(self, frame_idx: int, dof_idx: int, delta: float) -> None

    # 区间编辑
    def interpolate_segment(self, i0: int, i1: int,
                             method: str = 'linear') -> None
    def smooth_segment(self, i0: int, i1: int,
                       method: str = 'savgol',
                       window: int = 11, polyorder: int = 3) -> None

    # 跨帧传播
    def propagate(self, frame_idx: int, field: str,
                  delta: np.ndarray, n_frames: int,
                  decay: str = 'linear') -> None
    """将 frame_idx 的 delta 以衰减方式叠加到后续 n_frames 帧。
    
    decay 选项：'linear'（线性衰减到 0）、'cosine'（余弦衰减）、
                'constant'（等量叠加，不衰减）
    """

    # 撤销 / 重做
    def undo(self) -> bool
    def redo(self) -> bool

    # IK 编辑（P2，需要 pinocchio）
    def edit_end_effector(self, frame_idx: int,
                          ee_name: str, target_pos: np.ndarray) -> None
```

#### 跨帧传播算法

```
给定：当前帧 i，delta d，传播帧数 N，衰减函数 f(k) k∈[0,N-1]

for k in range(N):
    weight = f(k)
    motion[i + k][field] += weight * d

线性衰减：f(k) = 1 - k/(N-1)
余弦衰减：f(k) = 0.5 * (1 + cos(π * k / (N-1)))
常量：     f(k) = 1.0

编辑后对每帧执行：
  - root_rot：quat 归一化
  - dof_pos：joint limit clip
  - dof_vel：重新有限差分计算（若标志位开启）
```

### 3.6 IK 接口（Pinocchio-based，P2）

```python
class PinocchioIKSolver:
    """基于 Pinocchio 的末端 IK 求解器。
    
    可选依赖：`pip install motion-player[ik]` 安装 pinocchio。
    若 pinocchio 不可用，实例化时抛出 ImportError（优雅降级）。
    """
    def __init__(self, urdf_path: str | Path):
        """初始化 Pinocchio 模型（从 URDF）。"""

    def solve(self, q_init: np.ndarray,
              targets: dict[str, np.ndarray]) -> np.ndarray | None:
        """给定初始关节角和末端目标位置，返回解算后的关节角。
        
        targets: {end_effector_name: target_pos_xyz}
        返回 None 表示求解失败（未收敛）。
        """
```

### 3.7 MetricsEngine

**职责：** 可插拔质量指标计算，支持逐帧实时模式和全片段批量模式。

```python
class MetricTerm(ABC):
    """所有质量指标 term 的基类。"""
    name: str
    weight: float = 1.0

    @abstractmethod
    def compute_frame(self, motion: StandardMotion,
                      frame_idx: int) -> float:
        """计算单帧指标值（用于 HUD 实时显示）。"""

    @abstractmethod
    def compute_batch(self, motion: StandardMotion) -> np.ndarray:
        """计算全片段每帧的指标值，返回 shape (N_eff,) 的数组。"""


class MetricsEngine:
    def __init__(self, terms: list[MetricTerm] | None = None):
        """terms 为 None 时使用默认指标集（joint_limit + vel_spike + ...）。"""

    def register(self, term: MetricTerm) -> None

    def evaluate_frame(self, motion: StandardMotion,
                       frame_idx: int) -> dict[str, float]:
        """返回当前帧所有指标值 + composite score。"""

    def evaluate_batch(self, motion: StandardMotion) -> dict[str, np.ndarray]:
        """返回全片段每帧所有指标 + composite score 数组。"""

    def generate_report(self, motion: StandardMotion,
                        output_path: Path) -> None:
        """生成 JSON / CSV 质量报告。"""
```

#### 内置 MetricTerm 列表

| Term 类名 | 说明 | 对应 GMR 指标 |
|-----------|------|---------------|
| `JointLimitViolationTerm` | (frame, joint) 对中超限的比例 | `joint_limit_violation_rate` |
| `JointVelSpikeTerm` | 关节角速度峰值（有限差分） | `smoothness_penalty`（部分） |
| `JointAccelTerm` | 关节角加速度 RMS | `smoothness_penalty`（部分） |
| `RootLinVelTerm` | Root 线速度大小 | — |
| `RootAngVelTerm` | Root 角速度大小 | — |
| `SmoothnessTerm` | 关节速度方差均值 | `smoothness_penalty` |
| `FootPenetrationTerm` | 足端穿地检测（需要 site 定义） | — |
| `AMPFeatureStabilityTerm` | AMP 特征分布稳定性（均值/方差比） | — |
| `RootTrajectoryDTWTerm` | Root 轨迹 DTW（与参考对比） | `root_dtw_distance` |

### 3.8 Renderer 抽象（BaseRenderer）

```python
class BaseRenderer(ABC):
    """渲染后端的统一接口。"""

    @abstractmethod
    def load_model(self, robot_model: RobotModel) -> None:
        """加载机器人模型到渲染器。"""

    @abstractmethod
    def update_state(self, qpos: np.ndarray,
                     qvel: np.ndarray | None = None) -> None:
        """将 qpos/qvel 写入仿真状态并触发前向运动学。"""

    @abstractmethod
    def render_frame(self) -> None:
        """渲染一帧（含 HUD 叠加）。"""

    @abstractmethod
    def overlay_text(self, lines: list[str]) -> None:
        """在渲染窗口左上角叠加文字（用于质量 HUD）。"""

    @abstractmethod
    def close(self) -> None
```

### 3.9 MuJoCo 后端

```python
class MuJoCoRenderer(BaseRenderer):
    """MuJoCo kinematic replay 后端。
    
    使用 mujoco.viewer.launch_passive（推荐）或
    mujoco-python-viewer（兼容模式）。
    """
    def __init__(self, xml_path: str | Path,
                 camera_mode: str = 'follow_root'):
        """
        camera_mode:
          'follow_root' — 相机跟随 pelvis（第三人称）
          'fixed'       — 固定相机
          'free'        — 用户自由拖动
        """

    def set_ghost(self, motion: StandardMotion,
                  alpha: float = 0.3) -> None:
        """加载"幽灵"骨架（半透明叠加，用于 A/B 对比）。"""
```

**帧写入流程：**
```python
# 每帧更新
qpos = model_adapter.motion_to_qpos(motion, frame_idx)
# root: MuJoCo free joint = [x, y, z, qw, qx, qy, qz]
# 注意：MuJoCo 内部 quat 为 wxyz（scalar-first）
# DatasetAdapter 产出的 root_rot 为 xyzw，需要转换：
# mj_quat = [root_rot[3], root_rot[0], root_rot[1], root_rot[2]]
mj_data.qpos[:] = qpos
mj.mj_forward(mj_model, mj_data)
viewer.sync()
```

### 3.10 NVIDIA 后端（可选，P2）

```python
class NVHumanoidRenderer(BaseRenderer):
    """将 StandardMotion 适配到 NV ASE/CALM HumanoidViewMotion 格式。
    
    需要 IsaacGym 或 IsaacSim 环境（通过 extras[nv] 安装）。
    仅支持回放，不支持逐帧编辑。
    """
```

---

## 4. 配置 Schema 提案

### 4.1 `mapping.yaml`（关节映射配置）

```yaml
robot_mjcf_path: /path/to/assets/booster_t1/booster_t1.xml
root_joint_name: root        # free joint 名称
quat_convention: wxyz        # MuJoCo 内部四元数约定（通常固定为 wxyz）

# 数据集 DOF 顺序（可直接列出，也可指向 sidecar YAML）
dof_order_in_dataset:
  - left_hip_yaw
  - left_hip_roll
  # ... 

# 关节名称映射（数据集名 -> MJCF 名，若一致可省略）
name_map:
  left_hip_yaw: left_hip_yaw_joint

# 符号翻转（+1 或 -1）
sign_flip:
  right_hip_roll: -1.0

# 零位偏置（单位 rad）
offset:
  left_ankle_pitch: 0.0
```

### 4.2 `player_config.yaml`（播放器配置）

```yaml
# 数据集路径
dataset_path: /path/to/standard_motions/

# 机器人配置
mapping: /path/to/mapping.yaml

# 后端选择
backend: mujoco              # mujoco | nv
camera_mode: follow_root

# 质量评估
metrics:
  enabled: true
  weights:
    joint_limit_violation: 5.0
    joint_vel_spike: 0.5
    smoothness: 0.1
    root_dtw: 0.5

# 编辑配置
editing:
  undo_stack_size: 50
  propagation_decay: linear
  smooth_window: 11
  smooth_polyorder: 3
```

---

## 5. CLI 与交互设计

### 5.1 命令行接口

```bash
# 基础播放
motion-player play <motion_file.pkl> --mapping mapping.yaml

# 指定后端
motion-player play <file> --backend mujoco --camera follow_root

# 批量质量评估（无 GUI）
motion-player evaluate <dataset_dir> --mapping mapping.yaml --output report.json

# DOF 顺序审查
motion-player audit <motion_file.pkl> --robot-xml booster_t1.xml

# 生成 sidecar
motion-player gen-sidecar <motion_file.pkl> --robot booster_t1 --output meta.yaml
```

### 5.2 键盘快捷键（MuJoCo 窗口）

| 按键 | 功能 |
|------|------|
| `Space` | 播放 / 暂停 |
| `←` / `→` | 上一帧 / 下一帧 |
| `Shift+←` / `Shift+→` | 后退 / 前进 10 帧 |
| `Ctrl+←` / `Ctrl+→` | 后退 / 前进 100 帧 |
| `R` | 重置到第 0 帧 |
| `[` / `]` | 速度 ×0.5 / ×2 |
| `1` / `2` / ... | 切换 clip |
| `K` | 标记关键帧 |
| `I` / `O` | 设置区间起点 / 终点 |
| `S` | 对选定区间执行平滑 |
| `L` | 对选定区间执行线性插值 |
| `G` | 开关幽灵骨架（A/B 对比） |
| `E` | 导出当前编辑到文件 |
| `Q` | 打印当前帧质量指标 |
| `Ctrl+Z` | 撤销 |
| `Ctrl+Y` | 重做 |
| `H` | 显示帮助 |

---

## 6. 跨帧传播算法

### 6.1 线性衰减传播

```
weights[k] = max(0, 1 - k / n_frames),  k = 0, 1, ..., n_frames-1
motion[i+k].field += weights[k] * delta
```

### 6.2 余弦衰减传播

```
weights[k] = 0.5 * (1 + cos(π * k / n_frames))
motion[i+k].field += weights[k] * delta
```

### 6.3 约束投影（编辑后处理）

每次编辑（或传播）后，对受影响帧执行：

1. `root_rot` 归一化：`q /= ||q||`
2. `dof_pos` 限位裁剪：`dof_pos = clip(dof_pos, jnt_lo, jnt_hi)`（由 MJCF 读取）
3. 可选：重新计算 `dof_vel = diff(dof_pos) * fps`（若 `recompute_vel=True`）
4. 可选：重新计算 `root_lin_vel` / `root_ang_vel`（若 `recompute_root_vel=True`）

---

## 7. GMR Objective Reuse Mapping（GMR 优化目标复用映射）

> 本节基于对 `Mr-tooth/GMR` 代码库的探查，记录发现的优化目标 term，并与 MetricsEngine 接口进行映射。

### 7.1 已发现的 GMR 优化 Term

#### 来源文件：`general_motion_retargeting/benchmark/evaluator.py`

| GMR Term 名称 | 定义位置 | 计算方式 | 权重参数 |
|---|---|---|---|
| `ik_error` | `evaluator.py:RetargetingEvaluator._aggregate` | `mean(error1() + error2())` 跨所有帧 — `error1/2` 为各任务 tracking 误差的 L2 范数 | `EvaluatorWeights.ik_error`（默认 1.0） |
| `smoothness_penalty` | `evaluator.py:RetargetingEvaluator._aggregate` | `mean_j(var_t(Δq_j))`：各关节速度（有限差分）在时间维度的方差，再取关节均值 | `EvaluatorWeights.smoothness_penalty`（默认 0.1） |
| `joint_limit_violation_rate` | `evaluator.py:RetargetingEvaluator._compute_violation_rate` | 超出 MJCF `jnt_range` 的 (frame, joint) 对的比例 | `EvaluatorWeights.joint_limit_violation_rate`（默认 5.0） |
| `root_dtw_distance` | `evaluator.py:_dtw_distance` | DTW(robot_root_traj, human_pelvis_traj)，归一化（除以 T_a+T_b） | `EvaluatorWeights.root_dtw_distance`（默认 0.5） |

#### 来源文件：`general_motion_retargeting/benchmark/param_space.py`

| GMR 参数 | 类型 | 作用 | 与质量的关联 |
|---|---|---|---|
| `damping`（IK solver damping） | runtime 参数 | 控制 IK 求解阻尼，影响收敛和运动平滑度 | 间接影响 `ik_error` 和 `smoothness_penalty` |
| `human_scale_table[joint]` | per-joint scale | 缩放人体各关节段，影响重定向保真度 | 影响 `ik_error` 和 `root_dtw_distance` |
| `ik_match_table1/2[frame].pos_weight` | per-task weight | 位置 tracking 任务权重 | 影响 `ik_error` |
| `ik_match_table1/2[frame].rot_weight` | per-task weight | 旋转 tracking 任务权重 | 影响 `ik_error` |

#### 来源文件：`general_motion_retargeting/motion_retarget.py`

| GMR Term | 定义 | 说明 |
|---|---|---|
| `error1()` | `np.linalg.norm(concat([task.compute_error(cfg) for task in tasks1]))` | Stage-1 IK 任务（粗调：root + 主要肢体）的跟踪误差 |
| `error2()` | `np.linalg.norm(concat([task.compute_error(cfg) for task in tasks2]))` | Stage-2 IK 任务（精调：末端执行器等）的跟踪误差 |

#### Composite Score（GMR benchmark 综合目标）

```
composite_score = (
    w_ik * ik_error
    + w_smooth * smoothness_penalty
    + w_limit * joint_limit_violation_rate
    + w_root * root_dtw_distance
)
```

### 7.2 MetricsEngine 接口映射

| GMR Term | MetricTerm 类 | 是否需要额外输入 | 备注 |
|---|---|---|---|
| `ik_error` | **Gap**（见下） | 需要原始 IK 误差数据 | 标准数据已标准化，无法直接重现；需从 GMR 元数据中传入 |
| `smoothness_penalty` | `SmoothnessTerm` | 仅 dof_pos | 完全可从标准数据计算 |
| `joint_limit_violation_rate` | `JointLimitViolationTerm` | 需要 MJCF jnt_range | 可从 MuJoCo 模型读取 |
| `root_dtw_distance` | `RootTrajectoryDTWTerm` | 需要参考轨迹 | 若无参考，可与原始动作数据对比 |

### 7.3 Gap 与扩展点

**Gap 1：IK tracking error 无法从标准数据反向还原**
- GMR `ik_error` 依赖 retargeting 过程中的中间误差（IK solver 的任务误差）
- 标准数据仅包含最终 qpos，不含 IK 中间状态
- **扩展点**：MetricsEngine 支持接收外部传入的 `ik_error_per_frame` 数组（从 GMR 导出时附带）；若无则该 term 返回 NaN 并提示

**Gap 2：单帧质量 vs. 序列质量**
- GMR evaluator 面向序列（整段动作）
- motion-player 需要**逐帧**实时显示
- **扩展点**：所有 term 同时实现 `compute_frame` 和 `compute_batch`，batch 模式与 GMR evaluator 兼容

**Gap 3：AMP 训练友好性尚无 GMR 对应 term**
- GMR 当前无 "AMP feature distribution stability" 指标
- **提案**：新增 `AMPFeatureStabilityTerm`，计算 AMP discriminator input 特征（`root_pos`, `root_rot`, `dof_pos`, `key_body_pos_local`）的时序统计（均值 / 方差 / 帧间差分 L2）

**Gap 4：足端接触一致性**
- GMR 未定义 foot penetration 指标
- **提案**：`FootPenetrationTerm` 需要 foot site 名称配置，从 MJCF 或 sidecar 读取

---

## 8. 打包策略

### 8.1 目录结构

```
motion_player/          # Python 包根目录（在 GMR repo 顶层）
pyproject.toml          # standalone 打包元数据（motion-player 包）
README_motion_player.md
```

### 8.2 `pyproject.toml` extras 分组

```toml
[project.optional-dependencies]
ik = ["pin"]                          # Pinocchio（IK 编辑）
nv = ["isaacgym"]                     # NVIDIA 后端（用户自行安装）
dev = ["pytest", "ruff"]
```

### 8.3 下游集成契约

下游项目（`rsl-rl-ex`、`LeggedLabUltra`）通过以下方式集成：

```python
# 方式 1：programmatic API
from motion_player import play_motion, evaluate_motion
from motion_player.core.models import StandardMotion
from motion_player.core.adapters import DatasetAdapter

motion = DatasetAdapter().load("clip.pkl")
report = evaluate_motion(motion, mapping_config="mapping.yaml")

# 方式 2：CLI
# motion-player play clip.pkl --mapping mapping.yaml
```

---

## 9. 测试策略

| 测试层级 | 内容 | 工具 |
|---|---|---|
| 单元测试 | `StandardMotion` dataclass、`DatasetAdapter` 加载/保存、`EditingEngine` 各操作 | pytest |
| 集成测试 | `ModelAdapter` + `MuJoCoRenderer` 端到端（headless MuJoCo） | pytest + mujoco offscreen |
| 黄金文件测试 | 标准数据的质量报告输出（JSON）与预期对比 | pytest + numpy assert |
| 视觉冒烟测试 | 截图对比（可选，CI 中跳过 GPU 测试） | 手动 / imageio |

---

## 10. 开放问题清单

| # | 问题 | 状态 | 负责方向 |
|---|------|------|---------|
| 1 | Booster T1 MJCF 路径在 motion-player 中如何配置？是否从 LeggedLabUltra assets 复制一份？ | 开放 | 用户配置 mapping.yaml |
| 2 | `key_body_pos_local` 的 body 列表是否固定？还是每个 clip 可能不同？ | 开放 | 建议通过 sidecar YAML 固定 |
| 3 | Pinocchio 在 Windows 上的安装方案（conda-forge 是否足够？） | 开放 | 评估 conda-forge `pinocchio` |
| 4 | NV backend 是否需要支持 Isaac Lab（新版）还是仅 IsaacGym（旧版）？ | 开放 | 用户确认 |
| 5 | Ghost overlay 在 MuJoCo 中的实现方式：加载第二个模型实例还是修改 geom alpha？ | 开放 | 技术评估 |
| 6 | DOF 顺序审查工具是否需要支持自动修复（in-place reorder）或仅报告？ | 开放 | MVP 仅报告，V1 修复 |
| 7 | AMP feature stability term 的具体数学定义需要与 LeggedLabUltra discriminator 对齐 | 开放 | 等待 discriminator 代码参考 |
