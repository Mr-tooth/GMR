"""
Retargeting Quality Evaluator for GMR Benchmark.

Computes the following metrics over a batch of motion sequences:

1. **IK tracking error** - mean of ``error1()`` and ``error2()`` across all
   frames; reflects how closely the robot body matches the target human body.
2. **Smoothness penalty** - mean across joints of per-joint angular-velocity
   variance (finite difference of successive ``qpos`` frames); penalises jerky
   output.  Computed as ``mean_j(var_t(Δq_j))`` so that the metric is
   independent of sequence length and number of DOFs.
3. **Joint-limit violation rate** - fraction of (frame, joint) pairs where the
   joint angle lies outside the MJCF-defined ``jnt_range``; a value > 0
   indicates physically infeasible poses.
4. **Root trajectory DTW distance** - Dynamic Time Warping distance between
   the normalised robot root trajectory and the normalised human pelvis
   trajectory; captures global motion fidelity.  The DTW matrix is computed
   row-by-row with NumPy for efficiency.

The **composite score** (minimised during optimisation) is::

    score = (w_ik * ik_error
             + w_smooth * smoothness_penalty
             + w_limit * joint_limit_violation_rate
             + w_root * root_dtw_distance)

All weights are configurable via :class:`EvaluatorWeights`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Frame = dict  # {joint_name: (np.ndarray[3], np.ndarray[4])}
SequenceData = tuple[list[Frame], float]  # (frames, human_height)


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

    Uses a NumPy row-by-row DP that avoids pure-Python nested loops.
    For sequences of length T the pure-Python version requires T² Python
    iterations; the vectorised version processes one row at a time using
    NumPy broadcast operations, which is typically 10-50x faster.

    Parameters
    ----------
    seq_a : np.ndarray
        Array of shape ``(T_a, D)``.
    seq_b : np.ndarray
        Array of shape ``(T_b, D)``.

    Returns
    -------
    float
        Normalised DTW distance (divided by ``T_a + T_b`` to make it
        length-independent).
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0

    # --- pre-compute all pairwise L2 costs: shape (n, m) ---
    # seq_a[:, None, :] broadcasts over m; seq_b[None, :, :] over n
    cost_matrix = np.linalg.norm(seq_a[:, None, :] - seq_b[None, :, :], axis=-1)

    # --- row-by-row DP; only two rows in memory at a time ---
    # prev_row[j] = DTW(seq_a[:i], seq_b[:j+1])
    prev_row = np.full(m, np.inf)
    prev_row[0] = cost_matrix[0, 0]
    for j in range(1, m):
        # First row: only left neighbour is valid
        prev_row[j] = cost_matrix[0, j] + prev_row[j - 1]

    for i in range(1, n):
        curr_row = np.empty(m)
        # First column: only top neighbour is valid
        curr_row[0] = cost_matrix[i, 0] + prev_row[0]
        for j in range(1, m):
            # Standard DTW recurrence: min(left, top, diagonal) + cost
            best = min(curr_row[j - 1], prev_row[j], prev_row[j - 1])
            curr_row[j] = cost_matrix[i, j] + best
        prev_row = curr_row

    return float(prev_row[m - 1]) / (n + m)


# ---------------------------------------------------------------------------
# RetargetingEvaluator
# ---------------------------------------------------------------------------

class RetargetingEvaluator:
    """Evaluate retargeting quality for a given IK config.

    The MuJoCo robot model is loaded once in ``__init__`` and cached for
    the lifetime of the evaluator, avoiding repeated XML parsing across
    trials (Bug 2 fix).

    Parameters
    ----------
    src_human : str
        Human motion source identifier (e.g. ``'bvh_lafan1'``, ``'smplx'``).
    tgt_robot : str
        Robot identifier (e.g. ``'unitree_g1'``, ``'booster_t1'``).
    weights : EvaluatorWeights, optional
        Controls the composite objective weights.  Uses default weights
        if ``None``.
    verbose : bool
        Print per-sequence progress if ``True``.
    """

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        weights: EvaluatorWeights | None = None,
        verbose: bool = False,
    ) -> None:
        self.src_human = src_human
        self.tgt_robot = tgt_robot
        self.weights = weights if weights is not None else EvaluatorWeights()
        self.verbose = verbose

        # --- cache the MuJoCo model and joint-limit index once (Bug 2 fix) ---
        # Precompute (qpos_address, lower_bound, upper_bound) for every
        # limited non-free joint so that _compute_violation_rate is O(1)
        # on model loading per evaluator instance instead of O(trials).
        self._ranged_indices: list[tuple[int, float, float]] = []
        self._load_joint_limits()

    def _load_joint_limits(self) -> None:
        """Load the robot's joint limits from MJCF and cache them.

        Populates ``self._ranged_indices`` with ``(qpos_adr, lo, hi)``
        tuples for every limited non-free joint.  Silently leaves the list
        empty if the robot XML cannot be found (e.g. in unit tests that do
        not have access to robot assets).
        """
        try:
            from general_motion_retargeting.params import ROBOT_XML_DICT
            model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[self.tgt_robot]))
        except Exception:
            # Robot model unavailable; violation rate will always be 0.0
            return

        for jnt_id in range(model.njnt):
            # Only check joints with explicit range limits
            if not model.jnt_limited[jnt_id]:
                continue
            # Free joint (type 0) covers the 6-DOF root - skip it
            if model.jnt_type[jnt_id] == 0:
                continue
            qpos_adr = model.jnt_qposadr[jnt_id]
            lo, hi = model.jnt_range[jnt_id]
            self._ranged_indices.append((int(qpos_adr), float(lo), float(hi)))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sequences: list[SequenceData],
        ik_config_override: dict,
        damping: float = 0.5,
    ) -> dict[str, float]:
        """Run retargeting on all *sequences* and return quality metrics.

        Parameters
        ----------
        sequences : list of (frames, human_height)
            Motion sequences as returned by
            :class:`~general_motion_retargeting.benchmark.dataset_loader.DatasetLoader`.
        ik_config_override : dict
            Modified IK config dict produced by
            :meth:`~general_motion_retargeting.benchmark.param_space.IKConfigParamSpace.build_config`.
        damping : float
            IK solver damping value passed directly to
            ``GeneralMotionRetargeting`` (not stored in the config dict).

        Returns
        -------
        dict
            Keys: ``'ik_error'``, ``'smoothness_penalty'``,
            ``'joint_limit_violation_rate'``, ``'root_dtw_distance'``,
            ``'composite_score'``, ``'n_sequences'``, ``'n_frames_total'``.
        """
        # Accumulate per-sequence data for later aggregation
        all_ik_errors: list[float] = []
        all_qpos: list[np.ndarray] = []  # one (T, nq) array per sequence
        all_root_robot: list[np.ndarray] = []  # one (T, 3) array per sequence
        all_root_human: list[np.ndarray] = []  # one (T, 3) array per sequence

        for seq_idx, (frames, human_height) in enumerate(sequences):
            # Skip degenerate (empty) sequences
            if not frames:
                continue

            # --- initialise the retargeter for this sequence ---
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

            # Per-sequence accumulators
            seq_ik_errors: list[float] = []
            seq_qpos: list[np.ndarray] = []
            seq_root_robot: list[np.ndarray] = []
            seq_root_human: list[np.ndarray] = []

            # --- process every frame in the sequence ---
            for frame in frames:
                try:
                    # deep-copy so the retargeter cannot modify the cached frame
                    qpos = retargeter.retarget(copy.deepcopy(frame))
                    ik_err = retargeter.error1() + retargeter.error2()
                    seq_ik_errors.append(ik_err)
                    seq_qpos.append(qpos.copy())
                    # root position = first 3 elements of qpos (free joint XYZ)
                    seq_root_robot.append(qpos[:3].copy())
                    # human pelvis position from the scaled (height-adjusted) data
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

            # Only append if at least one frame succeeded
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

            # Explicitly free MuJoCo resources held by the retargeter
            del retargeter

        # Aggregate all per-sequence data into a single metrics dict
        metrics = self._aggregate(all_ik_errors, all_qpos, all_root_robot, all_root_human)
        return metrics

    # ------------------------------------------------------------------
    # Internal metric computation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        all_ik_errors: list[float],
        all_qpos: list[np.ndarray],
        all_root_robot: list[np.ndarray],
        all_root_human: list[np.ndarray],
    ) -> dict[str, float]:
        """Aggregate per-frame metrics into a single metrics dict.

        Parameters
        ----------
        all_ik_errors : list of float
            IK error (error1 + error2) for every processed frame.
        all_qpos : list of np.ndarray
            Per-sequence qpos arrays of shape ``(T, nq)``.
        all_root_robot : list of np.ndarray
            Per-sequence robot root positions, shape ``(T, 3)``.
        all_root_human : list of np.ndarray
            Per-sequence human pelvis positions, shape ``(T, 3)``.

        Returns
        -------
        dict
            Metrics dict with seven keys; see :meth:`evaluate`.
        """
        n_frames_total = sum(len(q) for q in all_qpos)

        # --- 1. IK tracking error ---
        # Mean of (error1 + error2) across all frames; lower is better.
        ik_error = float(np.mean(all_ik_errors)) if all_ik_errors else float("inf")

        # --- 2. Smoothness penalty ---
        # Compute the mean-over-joints of per-joint velocity variance:
        #   var_t(Δq_j) for each joint j, then average over j.
        # This is invariant to sequence length T and DOF count, unlike
        # np.var(joint_vel) which conflates both dimensions (Bug 1 fix).
        smooth_vals: list[float] = []
        for qpos_seq in all_qpos:
            if len(qpos_seq) < 2:
                continue
            # finite-difference velocity: shape (T-1, nq)
            vel = np.diff(qpos_seq, axis=0)
            # skip the 7 free-joint DOFs (3 translation + 4 quaternion)
            joint_vel = vel[:, 7:] if qpos_seq.shape[1] > 7 else vel
            # var per joint over time, then mean over joints
            smooth_vals.append(float(np.mean(np.var(joint_vel, axis=0))))
        smoothness_penalty = float(np.mean(smooth_vals)) if smooth_vals else 0.0

        # --- 3. Joint-limit violation rate ---
        # Uses the cached joint-limit index populated in __init__
        violation_rate = self._compute_violation_rate(all_qpos)

        # --- 4. Root trajectory DTW distance ---
        # Normalise each trajectory to start at the origin to remove the
        # absolute position offset before measuring trajectory similarity.
        dtw_vals: list[float] = []
        for robot_traj, human_traj in zip(all_root_robot, all_root_human, strict=True):
            if len(robot_traj) < 2:
                continue
            r_norm = robot_traj - robot_traj[0]
            h_norm = human_traj - human_traj[0]
            dtw_vals.append(_dtw_distance(r_norm, h_norm))
        root_dtw_distance = float(np.mean(dtw_vals)) if dtw_vals else 0.0

        # --- composite score (minimised during Optuna search) ---
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

    def _compute_violation_rate(self, all_qpos: list[np.ndarray]) -> float:
        """Return the fraction of (frame, joint) pairs that violate joint limits.

        Uses the pre-cached ``_ranged_indices`` from ``_load_joint_limits``
        so that no MuJoCo XML parsing occurs here (Bug 2 fix).

        Parameters
        ----------
        all_qpos : list of np.ndarray
            Per-sequence qpos arrays of shape ``(T, nq)``.

        Returns
        -------
        float
            Violation fraction in ``[0, 1]``.  Returns ``0.0`` if the
            cached index is empty or no sequences were provided.
        """
        if not self._ranged_indices or not all_qpos:
            return 0.0

        total_checks = 0
        total_violations = 0
        for qpos_seq in all_qpos:
            for qpos in qpos_seq:
                for adr, lo, hi in self._ranged_indices:
                    total_checks += 1
                    val = float(qpos[adr])
                    # Count any angle outside the [lo, hi] range as a violation
                    if val < lo or val > hi:
                        total_violations += 1

        return float(total_violations) / float(total_checks) if total_checks > 0 else 0.0
