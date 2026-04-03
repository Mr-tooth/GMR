"""
Retargeting Quality Evaluator for GMR Benchmark.

Computes the following metrics over a batch of motion sequences:

1. **IK tracking error** – mean of ``error1()`` and ``error2()`` across all
   frames; reflects how closely the robot body matches the target human body.
2. **Smoothness penalty** – variance of per-joint velocity (finite difference
   of successive ``qpos`` frames); penalises jerky output.
3. **Joint-limit violation rate** – fraction of (frame, joint) pairs where the
   joint angle lies outside the MJCF-defined ``jnt_range``; a value > 0
   indicates physically infeasible poses.
4. **Root trajectory DTW distance** – Dynamic Time Warping distance between
   the normalised robot root trajectory and the normalised human pelvis
   trajectory; captures global motion fidelity.

The **composite score** (minimised during optimisation) is::

    score = (w_ik * ik_error
             + w_smooth * smoothness_penalty
             + w_limit * joint_limit_violation_rate
             + w_root * root_dtw_distance)

All weights are configurable via :class:`EvaluatorWeights`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco as mj
import numpy as np

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Frame = dict  # {joint_name: (np.ndarray[3], np.ndarray[4])}
SequenceData = Tuple[List[Frame], float]  # (frames, human_height)


# ---------------------------------------------------------------------------
# Weights dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorWeights:
    """Weights for the composite optimisation objective.

    All weights must be ≥ 0.  Setting a weight to 0 disables the
    corresponding metric in the composite score (but it is still computed
    and reported in the detailed metrics dict).
    """
    ik_error: float = 1.0
    smoothness_penalty: float = 0.1
    joint_limit_violation_rate: float = 5.0
    root_dtw_distance: float = 0.5


# ---------------------------------------------------------------------------
# DTW helper (pure NumPy, no extra dependency)
# ---------------------------------------------------------------------------

def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Compute the DTW distance between two sequences of vectors.

    Parameters
    ----------
    seq_a, seq_b:
        Arrays of shape ``(T_a, D)`` and ``(T_b, D)``.

    Returns
    -------
    float
        Normalised DTW distance (divided by ``T_a + T_b`` to make it length-
        independent).
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0

    # Use flat array for efficiency
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m]) / (n + m)


# ---------------------------------------------------------------------------
# RetargetingEvaluator
# ---------------------------------------------------------------------------

class RetargetingEvaluator:
    """Evaluate retargeting quality for a given IK config.

    Parameters
    ----------
    src_human:
        Human motion source identifier (e.g. ``'bvh_lafan1'``, ``'smplx'``).
    tgt_robot:
        Robot identifier (e.g. ``'unitree_g1'``, ``'booster_t1'``).
    weights:
        Instance of :class:`EvaluatorWeights` controlling the composite
        objective.  If ``None``, default weights are used.
    verbose:
        Print per-sequence progress if ``True``.
    """

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        weights: Optional[EvaluatorWeights] = None,
        verbose: bool = False,
    ) -> None:
        self.src_human = src_human
        self.tgt_robot = tgt_robot
        self.weights = weights if weights is not None else EvaluatorWeights()
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sequences: List[SequenceData],
        ik_config_override: dict,
        damping: float = 0.5,
    ) -> Dict[str, float]:
        """Run retargeting on all *sequences* and return quality metrics.

        Parameters
        ----------
        sequences:
            List of ``(frames, human_height)`` tuples as returned by
            :class:`~general_motion_retargeting.benchmark.dataset_loader.DatasetLoader`.
        ik_config_override:
            Modified IK config dict to use (produced by
            :meth:`~general_motion_retargeting.benchmark.param_space.IKConfigParamSpace.build_config`).
        damping:
            IK solver damping value (separate from the config dict because it
            is passed directly to ``GeneralMotionRetargeting``).

        Returns
        -------
        dict
            Keys: ``'ik_error'``, ``'smoothness_penalty'``,
            ``'joint_limit_violation_rate'``, ``'root_dtw_distance'``,
            ``'composite_score'``, ``'n_sequences'``, ``'n_frames_total'``.
        """
        all_ik_errors: List[float] = []
        all_qpos: List[np.ndarray] = []  # each element is qpos for one sequence
        all_root_robot: List[np.ndarray] = []
        all_root_human: List[np.ndarray] = []

        for seq_idx, (frames, human_height) in enumerate(sequences):
            if not frames:
                continue
            try:
                retargeter = GeneralMotionRetargeting(
                    src_human=self.src_human,
                    tgt_robot=self.tgt_robot,
                    actual_human_height=human_height,
                    damping=damping,
                    verbose=False,
                    ik_config_override=ik_config_override,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"  [WARN] Failed to init retargeter for seq {seq_idx}: {exc}")
                continue

            seq_ik_errors: List[float] = []
            seq_qpos: List[np.ndarray] = []
            seq_root_robot: List[np.ndarray] = []
            seq_root_human: List[np.ndarray] = []

            for frame in frames:
                try:
                    qpos = retargeter.retarget(copy.deepcopy(frame))
                    ik_err = retargeter.error1() + retargeter.error2()
                    seq_ik_errors.append(ik_err)
                    seq_qpos.append(qpos.copy())
                    # root position: first 3 elements of qpos (free joint)
                    seq_root_robot.append(qpos[:3].copy())
                    # human root position from scaled human data
                    human_root = retargeter.human_root_name
                    if human_root in retargeter.scaled_human_data:
                        seq_root_human.append(
                            retargeter.scaled_human_data[human_root][0].copy()
                        )
                    else:
                        seq_root_human.append(np.zeros(3))
                except Exception as exc:
                    if self.verbose:
                        print(f"  [WARN] Frame retarget failed in seq {seq_idx}: {exc}")
                    continue

            if seq_qpos:
                all_ik_errors.extend(seq_ik_errors)
                all_qpos.append(np.stack(seq_qpos))
                all_root_robot.append(np.stack(seq_root_robot))
                all_root_human.append(np.stack(seq_root_human))

            if self.verbose:
                mean_err = float(np.mean(seq_ik_errors)) if seq_ik_errors else float("nan")
                print(
                    f"  seq {seq_idx}: {len(seq_qpos)} frames, "
                    f"mean IK error={mean_err:.4f}"
                )

            # Free retargeter resources explicitly (MuJoCo model lives here)
            del retargeter

        # Aggregate over all sequences
        metrics = self._aggregate(all_ik_errors, all_qpos, all_root_robot, all_root_human)
        return metrics

    # ------------------------------------------------------------------
    # Internal metric computation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        all_ik_errors: List[float],
        all_qpos: List[np.ndarray],
        all_root_robot: List[np.ndarray],
        all_root_human: List[np.ndarray],
    ) -> Dict[str, float]:
        """Aggregate per-frame metrics into sequence-level statistics."""

        n_frames_total = sum(len(q) for q in all_qpos)

        # 1. IK error
        ik_error = float(np.mean(all_ik_errors)) if all_ik_errors else float("inf")

        # 2. Smoothness penalty (mean variance of joint velocity across sequences)
        smooth_vals: List[float] = []
        for qpos_seq in all_qpos:
            if len(qpos_seq) < 2:
                continue
            vel = np.diff(qpos_seq, axis=0)  # shape (T-1, nq)
            # Use only the articulated joint part (skip free joint: first 7)
            joint_vel = vel[:, 7:] if qpos_seq.shape[1] >= 7 else vel
            smooth_vals.append(float(np.var(joint_vel)))
        smoothness_penalty = float(np.mean(smooth_vals)) if smooth_vals else 0.0

        # 3. Joint-limit violation rate
        violation_rate = self._compute_violation_rate(all_qpos)

        # 4. Root trajectory DTW distance
        dtw_vals: List[float] = []
        for robot_traj, human_traj in zip(all_root_robot, all_root_human):
            if len(robot_traj) < 2:
                continue
            # Normalise trajectories to start at origin (remove translation bias)
            r_norm = robot_traj - robot_traj[0]
            h_norm = human_traj - human_traj[0]
            dtw_vals.append(_dtw_distance(r_norm, h_norm))
        root_dtw_distance = float(np.mean(dtw_vals)) if dtw_vals else 0.0

        # Composite score (minimised)
        w = self.weights
        composite_score = (
            w.ik_error * ik_error
            + w.smoothness_penalty * smoothness_penalty
            + w.joint_limit_violation_rate * violation_rate
            + w.root_dtw_distance * root_dtw_distance
        )

        return {
            "ik_error": ik_error,
            "smoothness_penalty": smoothness_penalty,
            "joint_limit_violation_rate": violation_rate,
            "root_dtw_distance": root_dtw_distance,
            "composite_score": composite_score,
            "n_sequences": len(all_qpos),
            "n_frames_total": n_frames_total,
        }

    def _compute_violation_rate(self, all_qpos: List[np.ndarray]) -> float:
        """Return fraction of (frame, joint) pairs violating joint limits.

        Uses the MuJoCo model's ``jnt_range`` to check limits.  Only revolute
        and prismatic joints are checked (the free-joint DOFs 0–6 are skipped).
        Returns 0.0 if the model cannot be loaded or has no ranged joints.
        """
        try:
            from general_motion_retargeting.params import ROBOT_XML_DICT
            model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[self.tgt_robot]))
        except Exception:
            return 0.0

        # Collect joint ranges for articulated joints (skip free joint at index 0)
        ranged_indices: List[Tuple[int, float, float]] = []
        for jnt_id in range(model.njnt):
            if not model.jnt_limited[jnt_id]:
                continue
            jnt_type = model.jnt_type[jnt_id]
            # Free joint type = 0, slide = 2, hinge = 3
            if jnt_type == 0:  # free joint – skip
                continue
            qpos_adr = model.jnt_qposadr[jnt_id]
            lo, hi = model.jnt_range[jnt_id]
            ranged_indices.append((qpos_adr, float(lo), float(hi)))

        if not ranged_indices or not all_qpos:
            return 0.0

        total_checks = 0
        total_violations = 0
        for qpos_seq in all_qpos:
            for qpos in qpos_seq:
                for adr, lo, hi in ranged_indices:
                    total_checks += 1
                    val = float(qpos[adr])
                    if val < lo or val > hi:
                        total_violations += 1

        return float(total_violations) / float(total_checks) if total_checks > 0 else 0.0
