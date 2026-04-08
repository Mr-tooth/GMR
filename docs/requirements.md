# Motion Dataset Player / Editor — 需求文档 (PRD)

> 版本：v0.1  
> 状态：草稿  
> 最后更新：2025-04

---

## 1. 背景与目标

### 1.1 背景

用户已完成以下完整 pipeline：

1. **GMR 重定向**：LAFAN1 BVH → Booster T1 机器人动作（`bvh_to_robot.py`）
2. **标准化**：通过 `rsl-rl-ex` 的 `dataset_builder` 将 GMR 输出的 `.pkl` 转为 AMP `motion_loader` 格式的标准数据集
3. **下游验证**：在 `LeggedLabUltra` 中完成了初步的 AMP 训练测试

现在需要一个独立工具，能够：

- 在目标机器人（Booster T1 等）上**回放**标准化后的机器人动作数据集
- 支持**逐帧检查**与**人工微调**轨迹
- 输出**质量评估报告**（与 GMR 优化目标对齐）

### 1.2 目标

| 目标 | 描述 |
|------|------|
| **P0** 回放 | 能够在 MuJoCo 窗口中播放标准数据集里的机器人动作 |
| **P0** 质量检查 | 逐帧显示 AMP 相关质量指标（joint limit、速度尖峰等） |
| **P1** 逐帧编辑 | 支持对单帧的 root 姿态、DOF 角度进行增量调整 |
| **P1** 区间编辑 | 对选定区间做插值平滑或滤波修复 |
| **P2** IK 编辑 | 通过末端位置反求关节角（Pinocchio-based） |
| **P2** 多后端 | 兼容 NVIDIA HumanoidViewMotion 作为可选回放后端 |

### 1.3 项目定位

该工具作为**独立 Python 包**（`motion-player`）开发，可被下游项目（`rsl-rl-ex`、`LeggedLabUltra`）通过 `pip install` 引入，也可独立运行。

---

## 2. 范围界定

### 2.1 IN SCOPE（本工具包含）

- 标准数据集（`*_standard.pkl` / `.npy dict`）的加载与解析
- MuJoCo kinematic replay（运动学回放，不涉及物理仿真控制）
- 交互式播放控制（暂停、步进、时间轴拖拽）
- 逐帧手动编辑（root 平移/旋转、DOF 角度 delta）
- 区间平滑插值与修复
- 基于 Pinocchio 的末端 IK 编辑接口（P2）
- 质量指标 HUD（实时叠加显示）
- 质量报告导出（JSON / CSV）
- 视频录制导出（mp4）
- DOF 顺序审查与标准顺序修复模块（审查工具，非实时修复）
- 跨平台支持：Linux（优先）、Windows、macOS（渲染可较慢）
- 独立 pip 打包与安装

### 2.2 OUT OF SCOPE（本版本不包含）

- 物理仿真 replay（需要 PD 控制、力矩、碰撞响应）—— 计划 V2
- Isaac Gym / IsaacSim 的全功能适配（仅提供可选回放后端接口）
- URDF → MJCF 的自动转换（提供接口，依赖现有工具）
- 网页端 / 远程访问
- 多机器人同屏对比（超过 2 个）—— 计划 V2
- 真实感渲染（PhysX / Ray-tracing）

---

## 3. 用户画像与工作流

### 3.1 主要用户：机器人算法工程师（Persona A）

**典型工作流：**
1. 使用 GMR 将动捕数据重定向到机器人
2. 通过 `dataset_builder` 生成标准数据集
3. **打开 motion-player 可视化检查动作质量**
4. 发现有问题的片段，进行逐帧微调
5. 导出修复后的数据集继续 AMP 训练

### 3.2 次要用户：训练参数优化工程师（Persona B）

**典型工作流：**
1. 对同一组动捕数据跑多套 GMR 参数
2. **批量导入 motion-player，对比质量指标**
3. 根据 AMP 训练友好性指标选择最优参数

---

## 4. 功能需求

### 4.1 数据加载（FR-01）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-01-1 | 支持加载 `*_standard.pkl`（Python pickle，dict 格式） | P0 |
| FR-01-2 | 支持加载 `.npy` dict 格式（`np.load(..., allow_pickle=True)`） | P0 |
| FR-01-3 | 自动校验标准字段完整性（`fps`、`root_pos`、`root_rot`、`dof_pos` 必须存在） | P0 |
| FR-01-4 | 支持同时加载一个目录下的多个标准文件（dataset 模式） | P1 |
| FR-01-5 | 支持可选的 sidecar YAML（记录 DOF 名称/顺序，见 §4.8） | P1 |

### 4.2 机器人模型加载（FR-02）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-02-1 | 支持从 MJCF（`.xml`）加载机器人模型 | P0 |
| FR-02-2 | 支持从 URDF 加载（转换为 MuJoCo 兼容格式） | P1 |
| FR-02-3 | 自动建立 MJCF joint 顺序与数据集 DOF 列的映射（via sidecar 或启发式） | P0 |
| FR-02-4 | 支持通过 `mapping.yaml` 显式配置关节名称映射、符号翻转与偏置 | P0 |

### 4.3 播放控制（FR-03）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-03-1 | 播放 / 暂停（Space） | P0 |
| FR-03-2 | 逐帧步进 ±1 帧（← →） | P0 |
| FR-03-3 | 快进 / 快退 ±10 / ±100 帧（Shift+←/→） | P0 |
| FR-03-4 | 重置到第 0 帧（R） | P0 |
| FR-03-5 | 时间轴拖拽（scrub） | P1 |
| FR-03-6 | 播放速度倍率调节（0.25x / 0.5x / 1x / 2x） | P1 |
| FR-03-7 | 循环 / 乒乓模式 | P1 |
| FR-03-8 | 多 clip 切换（1 / 2 / 数字键） | P1 |

### 4.4 逐帧编辑（FR-04）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-04-1 | Root 平移增量调节（X/Y/Z 分量） | P1 |
| FR-04-2 | Root 旋转增量调节（Yaw / Pitch / Roll 三自由度均支持） | P1 |
| FR-04-3 | 单关节 DOF 角度增量调节（键盘快捷键 或 CLI 输入） | P1 |
| FR-04-4 | 编辑后自动做关节限位裁剪（joint limit clipping） | P1 |
| FR-04-5 | 编辑后自动做四元数归一化 | P1 |
| FR-04-6 | 撤销 / 重做（Ctrl+Z / Ctrl+Y，最多 50 步） | P1 |
| FR-04-7 | 末端 IK 编辑：拖动末端位置，反解关节角（Pinocchio-based） | P2 |

### 4.5 关键帧与区间编辑（FR-05）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-05-1 | 标记当前帧为关键帧（K） | P1 |
| FR-05-2 | 选定区间 `[i0, i1]` 做线性插值（root / DOF） | P1 |
| FR-05-3 | 选定区间做样条插值（Catmull-Rom） | P1 |
| FR-05-4 | 选定区间做平滑滤波（Savitzky–Golay 或低通） | P1 |
| FR-05-5 | 跨帧传播：对当前帧的 delta 以渐变方式叠加到后续 N 帧 | P1 |

### 4.6 质量评估（FR-06）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-06-1 | 实时 HUD 显示当前帧的关节限位违反数 | P0 |
| FR-06-2 | 实时 HUD 显示当前帧的关节速度峰值（有限差分） | P0 |
| FR-06-3 | 实时 HUD 显示 Root 线速度 / 角速度 | P0 |
| FR-06-4 | 全片段质量报告导出（JSON / CSV）：每帧各指标值 | P1 |
| FR-06-5 | 自动标记并高亮"质量最差"的 K 个区间 | P1 |
| FR-06-6 | AMP 特征分布统计（均值 / 方差 / 分位数）报告 | P1 |
| FR-06-7 | 接触一致性检测（足端是否穿地，需要足端 site 定义） | P2 |
| FR-06-8 | 与 GMR benchmark 指标对齐（IK 误差、平滑度惩罚、DTW） | P2 |

### 4.7 导出与持久化（FR-07）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-07-1 | 将编辑后的数据以相同标准格式保存为新 `.pkl` 文件 | P1 |
| FR-07-2 | 视频录制导出（mp4，1080p，可指定帧率） | P1 |
| FR-07-3 | 保存当前编辑状态（关键帧列表、标记区间）到 JSON sidecar | P1 |

### 4.8 DOF 顺序审查模块（FR-08）

| ID | 需求 | 优先级 |
|----|------|--------|
| FR-08-1 | 读取数据集文件并与 MJCF joint 顺序对比，输出差异报告 | P1 |
| FR-08-2 | 支持根据 sidecar YAML 中的 DOF 名称列表自动重排 DOF 列 | P1 |
| FR-08-3 | 为已有数据集生成 sidecar YAML（从 GMR 输出的元数据推断） | P1 |

---

## 5. 非功能需求

### 5.1 跨平台性

| 平台 | 目标 | 说明 |
|------|------|------|
| Linux | 完全支持，优先跑通 | 包括 Ubuntu 20.04 / 22.04 |
| Windows | 完全支持 | 需测试 MuJoCo / glfw 在 Windows 下的正常运行 |
| macOS | 支持（渲染可较慢） | 不要求 GPU 加速，可接受 CPU 软件渲染 |

### 5.2 可安装性

- 可通过 `pip install motion-player`（或 `pip install -e .`）安装
- 安装配置门槛低：核心依赖仅为 `mujoco`、`numpy`、`scipy`；可选依赖（Pinocchio、NV backend）通过 extras 分组
- 依赖齐全安装即可使用，不要求轻依赖

### 5.3 性能目标

| 指标 | 目标值 |
|------|--------|
| 回放帧率 | ≥ 30 FPS（Linux，CPU 渲染） |
| 单帧编辑响应延迟 | < 50 ms |
| 100 帧区间平滑 | < 500 ms |
| 全片段质量报告生成 | < 10 s / 1000 帧 |

### 5.4 可扩展性

- 后端（Renderer）抽象为插件接口，可独立实现新后端（NV、Web 等）
- 质量指标为可插拔 term 架构，支持自定义新指标
- 数据适配器（DatasetAdapter）支持扩展新的输入格式
- 编辑操作（EditOperation）为可序列化对象，支持脚本化批处理

---

## 6. 数据兼容需求

### 6.1 标准数据 Schema（来自 `rsl-rl-ex/utils/data_builder.py`）

| 字段 | Shape | 说明 |
|------|-------|------|
| `fps` | scalar | 帧率，默认 30 |
| `motion_length` | scalar | 有效帧数 = N−1（N-1 时间对齐） |
| `motion_weight` | scalar | clip 权重（用于采样） |
| `root_pos` | `(N-1, 3)` | Root 位置（t0 帧，world frame） |
| `root_rot` | `(N-1, 4)` | Root 旋转四元数，**scalar-last (xyzw)**（t0 帧） |
| `projected_gravity` | `(N-1, 3)` | 重力投影到 root-local 系 |
| `root_lin_vel` | `(N-1, 3)` | Root 线速度（root-local） |
| `root_ang_vel` | `(N-1, 3)` | Root 角速度（root-local） |
| `dof_pos` | `(N-1, num_dofs)` | 关节角度（t0 帧） |
| `dof_vel` | `(N-1, num_dofs)` | 关节角速度（有限差分） |
| `key_body_pos_local` | `(N-1, K×3)` | 所有 body 在 root-local 系的位置（flatten） |

**重要约定：**
- `root_rot` 在标准文件中为 **xyzw（scalar-last）**；`motion_loader` 支持转为 wxyz
- N-1 对齐语义：所有字段取 `[:-1]`（t0），速度类字段为相邻两帧差分
- 当前标准文件**不含** DOF 名称/顺序，由 sidecar YAML 补充

### 6.2 Sidecar YAML 提案（FR-08 支持）

```yaml
# <motion_clip_name>_meta.yaml
dof_names:
  - left_hip_yaw
  - left_hip_roll
  - left_hip_pitch
  # ...
key_body_names:
  - pelvis
  - left_foot
  - right_foot
  - left_hand
  - right_hand
  # ...（全部 body）
robot: booster_t1
source_pipeline: gmr_bvh_lafan1
gmr_ik_config: bvh_lafan1_to_t1_29dof
```

### 6.3 适配器扩展

DatasetAdapter 设计为插件式，未来支持：
- `rsl_rl_ex` 标准格式（当前版本）
- AMI-iit `amp-rsl-rl` 格式
- IsaacGym AMP 原始格式（直接 `.npy` dict with `dof_names` key）

---

## 7. 质量评估需求

### 7.1 优先级排序（用户明确）

1. **AMP 训练友好性**（最高优先级）
   - 特征分布稳定性（均值 / 方差 / 分位数）
   - joint limit 违反率（AMP discriminator 对此敏感）
   - 速度/加速度连续性（discriminator 对突变敏感）

2. **物理合理性**（次优先级）
   - 足端不穿地
   - COM 高度合理
   - 关节角度在 ROM 内

3. **视觉平滑 / 不抖动**（最低优先级）
   - 关节速度方差（平滑度惩罚）
   - Root 轨迹 DTW（与参考动作一致性）

### 7.2 指标 API 要求

- 每个指标为独立可插拔 `MetricTerm` 对象
- 支持加权线性组合得到 composite score
- composite score 与 GMR benchmark `EvaluatorWeights` 对齐（见 §8）
- 支持全片段批量计算与逐帧实时计算两种模式

---

## 8. 后端策略

### 8.1 MuJoCo 后端（主后端，P0）

- 默认且必选的渲染后端
- 使用 **kinematic replay 模式**：每帧直接写 `mjData.qpos`，调用 `mj_forward` 后渲染
- 交互窗口：`mujoco.viewer.launch_passive` 或 `mujoco-python-viewer` 库
- 支持自定义相机（跟随 pelvis 的第三人称视角）

### 8.2 NVIDIA HumanoidViewMotion 后端（可选，P2）

- 作为可选回放后端，**不要求**在 NV backend 中实现逐帧编辑
- 适配接口：将 `StandardMotion` 序列化为 NV ASE/CALM 的 `HumanoidViewMotion` 任务可识别的格式
- 需要 IsaacGym 或 IsaacSim 环境（通过 extras 依赖分组隔离）

---

## 9. 风险与缓解措施

| 风险 | 影响 | 缓解 |
|------|------|------|
| Booster T1 MJCF 在 motion-player 中不可访问 | 回放无法进行 | 在 `mapping.yaml` 中配置模型路径；支持传入任意 MJCF |
| DOF 顺序不匹配导致动作错乱 | 动作严重变形 | 提供 DOF 审查工具（FR-08）；警告提示 |
| Pinocchio 安装复杂（尤其 Windows） | IK 编辑功能不可用 | Pinocchio 作为可选依赖；IK 功能优雅降级 |
| macOS Metal/OpenGL 渲染问题 | macOS 无法运行 | 优先 Linux；macOS 渲染慢但可接受 |
| NV 后端与 IsaacGym 版本强耦合 | NV backend 不稳定 | NV backend 完全隔离为可选插件 |

---

## 10. 里程碑

### MVP（约 2-3 周）

- [ ] 加载标准 `.pkl`，在 MuJoCo 窗口中播放 Booster T1 动作
- [ ] 基础播放控制（Space / ← → / R）
- [ ] 实时 HUD：joint limit 违反数 + 关节速度峰值
- [ ] `mapping.yaml` 配置 DOF 映射

### V1（约 4-6 周）

- [ ] 完整播放控制（速度倍率、时间轴拖拽、多 clip 切换）
- [ ] 逐帧编辑（root 平移/旋转、DOF delta）
- [ ] 关键帧标记 + 区间插值/平滑
- [ ] 跨帧传播
- [ ] 质量报告导出（JSON / CSV）
- [ ] 视频录制
- [ ] DOF 顺序审查工具

### V1.5（约 8-10 周）

- [ ] Pinocchio IK 末端编辑
- [ ] NVIDIA HumanoidViewMotion 可选后端
- [ ] GMR benchmark 指标深度集成
- [ ] pip 发布（PyPI 或私有源）

---

## 11. 验收标准检查表

- [ ] 能够加载 `rsl-rl-ex` 生成的 `*_standard.pkl` 文件（至少含 Booster T1 数据集）
- [ ] 在 Linux 上能在 MuJoCo 窗口中流畅播放（≥ 30 FPS）
- [ ] 播放时 HUD 实时显示 joint limit 违反率
- [ ] 能对当前帧 root 旋转（yaw）做增量调整并立即刷新渲染
- [ ] 能对选定区间做 Savitzky–Golay 平滑并导出修复后文件
- [ ] 质量报告 JSON 包含每帧的 joint limit / vel_spike / root_vel 字段
- [ ] `pip install -e .` 在干净的 conda 环境（Python 3.10）中成功，且 `motion-player --help` 可运行
- [ ] 在 Windows 10 和 macOS 12 上完成冒烟测试（无 crash）
