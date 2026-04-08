# motion-player

> **Motion Dataset Player & Editor for Humanoid Robots**  
> A standalone tool for playing back, inspecting, and editing standardized robot motion datasets — MuJoCo-first, cross-platform.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

---

## Overview

**motion-player** is the interactive inspection and editing companion for the [GMR](../README.md) retargeting pipeline.  After retargeting human motion to a robot and standardizing it with [rsl-rl-ex](https://github.com/Mr-tooth/rsl-rl-ex), you can:

- 🎬 **Play back** your `*_standard.pkl` dataset on the actual robot in a MuJoCo window
- 🔍 **Inspect** quality metrics per frame (joint-limit violations, velocity spikes, AMP feature stability)
- ✏️ **Edit** individual frames (root pose, DOF angles) with undo/redo
- 📐 **Smooth** selected segments with Savitzky–Golay or low-pass filtering
- 📊 **Export** quality reports (JSON) and edited motions (`.pkl`)
- 🎥 **Record** playback videos (mp4)

---

## Quick Start

### 1. Install

```bash
# Clone GMR and install motion-player as a standalone package
cd /path/to/GMR

# Minimal install (no rendering — useful for headless evaluation)
pip install -e ".[dev]"

# With MuJoCo rendering (recommended)
pip install -e ".[mujoco,dev]"
```

> **Note:** `mujoco` is listed as an optional extra to keep the package lightweight.
> For the standard use case (interactive player), always install the `mujoco` extra.

### 2. Play a motion clip

```bash
motion-player play path/to/clip_standard.pkl \
    --mapping path/to/mapping.yaml
```

### 3. Evaluate quality (no GUI)

```bash
motion-player evaluate path/to/clip_standard.pkl \
    --mapping path/to/mapping.yaml \
    --output quality_report.json
```

### 4. Audit DOF order

```bash
motion-player audit path/to/clip_standard.pkl \
    --robot-xml path/to/booster_t1.xml
```

### 5. Generate DOF sidecar

```bash
motion-player gen-sidecar path/to/clip_standard.pkl \
    --robot-xml path/to/booster_t1.xml \
    --output path/to/clip_meta.yaml
```

### 6. Export to NVIDIA AMP format

```bash
motion-player convert-nv path/to/clip_standard.pkl \
    --output clip_nv.npy --wxyz
```

---

## Keyboard Shortcuts (MuJoCo Viewer)

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `←` / `→` | Previous / next frame |
| `Shift+←/→` | ±10 frames |
| `Ctrl+←/→` | ±100 frames |
| `R` | Reset to frame 0 |
| `[` / `]` | Speed ×0.5 / ×2 |
| `1`–`9` | Switch clip |
| `K` | Mark keyframe |
| `I` / `O` | Set segment start / end |
| `S` | Smooth selected segment |
| `L` | Interpolate selected segment |
| `G` | Toggle ghost overlay (A/B) |
| `E` | Export current edits |
| `Q` | Print frame quality metrics |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `H` | Help |

---

## mapping.yaml Example

Create a `mapping.yaml` to tell motion-player about your robot:

```yaml
robot_mjcf_path: /path/to/LeggedLabUltra/assets/booster_t1/booster_t1.xml
root_joint_name: root
quat_convention: wxyz    # MuJoCo internal convention (always wxyz)

# DOF order as produced by GMR (follows MuJoCo XML import order)
dof_order_in_dataset:
  - left_hip_yaw
  - left_hip_roll
  - left_hip_pitch
  - left_knee_pitch
  - left_ankle_pitch
  - left_ankle_roll
  - right_hip_yaw
  - right_hip_roll
  - right_hip_pitch
  - right_knee_pitch
  - right_ankle_pitch
  - right_ankle_roll
  # ... add remaining DOFs

# Optional: rename dataset DOF names to MJCF joint names
name_map: {}

# Optional: flip joint directions
sign_flip: {}

# Optional: zero-position offsets (radians)
offset: {}
```

---

## Data Schema Reference

The standard data format (from `rsl-rl-ex`):

| Field | Shape | Description |
|-------|-------|-------------|
| `fps` | scalar | Frame rate |
| `motion_length` | scalar | N−1 frames (AMP N-1 alignment) |
| `motion_weight` | scalar | Clip sampling weight |
| `root_pos` | `(N-1, 3)` | Root position (world frame) |
| `root_rot` | `(N-1, 4)` | Root quaternion **xyzw** |
| `projected_gravity` | `(N-1, 3)` | Gravity in root-local frame |
| `root_lin_vel` | `(N-1, 3)` | Root linear velocity (local) |
| `root_ang_vel` | `(N-1, 3)` | Root angular velocity (local) |
| `dof_pos` | `(N-1, D)` | Joint angles (radians) |
| `dof_vel` | `(N-1, D)` | Joint velocities (rad/s) |
| `key_body_pos_local` | `(N-1, K×3)` | All body positions (root-local) |

---

## Python API

```python
from motion_player import play_motion, evaluate_motion
from motion_player.core.adapters import DatasetAdapter, ModelAdapter
from motion_player.core.editing import EditingEngine
from motion_player.core.metrics import MetricsEngine

# Load a clip
motion = DatasetAdapter().load("clip_standard.pkl")

# Evaluate quality
engine = MetricsEngine()
report = engine.evaluate_batch(motion)
print(f"Joint limit violation rate (mean): {report['joint_limit_violation_rate'].mean():.4f}")

# Edit: smooth frames 10-50
editor = EditingEngine(motion)
editor.smooth_segment(10, 50, method="savgol", window=11)

# Save edited motion
DatasetAdapter().save(motion, "clip_edited.pkl")
```

---

## Architecture

```
motion_player/
├── core/
│   ├── models.py      # StandardMotion, RobotModel dataclasses
│   ├── adapters.py    # DatasetAdapter, ModelAdapter, DOFAuditor
│   ├── playback.py    # PlaybackEngine (frame state machine)
│   ├── editing.py     # EditingEngine (frame/segment edits, undo/redo)
│   └── metrics.py     # MetricsEngine + pluggable MetricTerm classes
├── backends/
│   ├── mujoco_backend.py  # MuJoCo kinematic replay + viewer
│   └── nv_backend.py      # NVIDIA HumanoidViewMotion (optional, V1.5)
└── cli.py             # Command-line interface
```

The `core/` modules have **no rendering dependencies** — they only need
`numpy` and `scipy`.  Rendering is handled entirely in `backends/`.

---

## Quality Metrics

motion-player implements the same quality objectives as the GMR benchmark
(`general_motion_retargeting/benchmark/evaluator.py`), extended with
AMP-training-specific terms:

| Metric | GMR Equivalent | Priority |
|--------|---------------|----------|
| `joint_limit_violation_rate` | ✓ `joint_limit_violation_rate` | P0 (AMP) |
| `joint_vel_spike` | ✓ `smoothness_penalty` (partial) | P0 (AMP) |
| `amp_feature_stability` | — (new) | P1 (AMP) |
| `smoothness_penalty` | ✓ `smoothness_penalty` | P2 |
| `root_lin_vel_magnitude` | — | informational |
| `root_ang_vel_magnitude` | — | informational |

See [`docs/design.md §7`](docs/design.md#7-gmr-objective-reuse-mapping) for the full GMR objective mapping.

---

## Roadmap

### MVP (weeks 1–3)
- [x] `StandardMotion` data model + `DatasetAdapter`
- [x] `PlaybackEngine` state machine
- [x] `EditingEngine` (frame edits, segment smooth, undo/redo)
- [x] `MetricsEngine` with default AMP terms
- [x] `MuJoCoRenderer` kinematic replay skeleton
- [x] CLI entry point
- [ ] **End-to-end play** on Booster T1 with real data
- [ ] Real-time HUD in MuJoCo viewer

### V1 (weeks 4–6)
- [ ] Full keybinding implementation in MuJoCo viewer
- [ ] Time-axis scrub bar
- [ ] Multi-clip switching
- [ ] Video recording export
- [ ] DOF audit & sidecar generation (complete)
- [ ] Quality report JSON/CSV

### V1.5 (weeks 8–10)
- [ ] Pinocchio IK end-effector editing
- [ ] NVIDIA HumanoidViewMotion backend
- [ ] PyPI publish

---

## Integration with Downstream Projects

```python
# In rsl-rl-ex or LeggedLabUltra:
# pip install git+https://github.com/Mr-tooth/GMR#subdirectory=.&egg=motion-player

from motion_player import evaluate_motion

report = evaluate_motion(
    "datasets/booster_t1_lafan1_standard.pkl",
    mapping_config="configs/booster_t1_mapping.yaml",
    output_path="reports/quality.json",
)
# Use report["joint_limit_violation_rate"] to filter out bad clips
# before AMP training.
```

---

## Contributing

This tool lives under `motion_player/` in the GMR repository.  To run the
existing tests:

```bash
cd /path/to/GMR
pytest tests/ -v
```

New tests for `motion_player` should be added to `tests/motion_player/`.

---

## License

MIT — see [LICENSE](../LICENSE).
