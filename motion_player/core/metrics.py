"""Quality metrics engine for motion_player.

This module provides a plug-in architecture for computing quality metrics
over :class:`~motion_player.core.models.StandardMotion` clips.

The metric system mirrors the objectives used in the GMR benchmark
(``general_motion_retargeting/benchmark/evaluator.py``) and extends them
with AMP-training-focused terms:

Priority order (per user requirements):
  1. AMP training friendliness  (joint limit violations, velocity continuity,
     feature distribution stability)
  2. Physical plausibility      (foot penetration, COM height)
  3. Visual smoothness / no jitter (smoothness penalty, DTW)

Usage
-----
::

    from motion_player.core.metrics import MetricsEngine

    engine = MetricsEngine()
    report = engine.evaluate_batch(motion)
    engine.generate_report(motion, output_path="quality_report.json")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from motion_player.core.models import RobotModel, StandardMotion

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Base MetricTerm
# ---------------------------------------------------------------------------

class MetricTerm(ABC):
    """Abstract base class for a single quality metric term.

    Subclasses must implement :meth:`compute_frame` (single-frame scalar)
    and :meth:`compute_batch` (full-clip per-frame array).

    Attributes
    ----------
    name : str
        Unique metric identifier (used as dict key in reports).
    weight : float
        Weight in the composite score (default 1.0).  Set to 0 to include
        in reports without contributing to the composite.
    """

    name: str = "unnamed"
    weight: float = 1.0

    @abstractmethod
    def compute_frame(
        self, motion: StandardMotion, frame_idx: int
    ) -> float:
        """Compute the metric value for a single frame.

        Parameters
        ----------
        motion:
            Source motion clip.
        frame_idx:
            Frame index (0-based).

        Returns
        -------
        float
            Scalar metric value.  ``float("nan")`` if the metric is not
            applicable for this frame.
        """

    @abstractmethod
    def compute_batch(self, motion: StandardMotion) -> np.ndarray:
        """Compute the metric for every frame in the clip.

        Parameters
        ----------
        motion:
            Source motion clip.

        Returns
        -------
        np.ndarray
            Shape ``(N_eff,)`` with one scalar per frame.  May contain
            ``nan`` for inapplicable frames.
        """


# ---------------------------------------------------------------------------
# Built-in metric terms
# ---------------------------------------------------------------------------

class JointLimitViolationTerm(MetricTerm):
    """Fraction of DOFs exceeding their joint limits in a frame.

    This corresponds to ``joint_limit_violation_rate`` in the GMR benchmark
    (``evaluator.py``), but computed per-frame rather than aggregated.

    A value > 0 indicates physically infeasible poses.  This is the most
    AMP-training-critical metric: the discriminator is highly sensitive to
    out-of-range joint angles.

    Parameters
    ----------
    robot_model:
        Robot model providing ``jnt_range``.  If ``None`` or
        ``jnt_range`` is unavailable, this term always returns 0.0.
    """

    name = "joint_limit_violation_rate"
    weight = 5.0  # mirror GMR EvaluatorWeights default

    def __init__(self, robot_model: RobotModel | None = None) -> None:
        self._robot_model = robot_model

    def compute_frame(
        self, motion: StandardMotion, frame_idx: int
    ) -> float:
        if (
            self._robot_model is None
            or self._robot_model.jnt_range is None
        ):
            return 0.0
        dof = motion.dof_pos[frame_idx]
        lo = self._robot_model.jnt_range[:, 0]
        hi = self._robot_model.jnt_range[:, 1]
        n_violations = int(np.sum((dof < lo) | (dof > hi)))
        return n_violations / max(len(dof), 1)

    def compute_batch(self, motion: StandardMotion) -> np.ndarray:
        if (
            self._robot_model is None
            or self._robot_model.jnt_range is None
        ):
            return np.zeros(motion.motion_length, dtype=np.float32)
        dof = motion.dof_pos  # (N, num_dofs)
        lo = self._robot_model.jnt_range[:, 0]
        hi = self._robot_model.jnt_range[:, 1]
        violations = ((dof < lo) | (dof > hi)).sum(axis=-1)  # (N,)
        return (violations / max(motion.num_dofs, 1)).astype(np.float32)


class JointVelSpikeTerm(MetricTerm):
    """Maximum absolute joint velocity in a frame (rad/s).

    Detects abrupt jitter — values much larger than typical motion speed
    indicate data artefacts that will confuse the AMP discriminator.

    Parameters
    ----------
    threshold:
        Value above which the metric is considered a "spike".
        Default ``10.0`` rad/s (adjustable).
    """

    name = "joint_vel_spike"
    weight = 0.5

    def __init__(self, threshold: float = 10.0) -> None:
        self._threshold = threshold

    def compute_frame(
        self, motion: StandardMotion, frame_idx: int
    ) -> float:
        return float(np.max(np.abs(motion.dof_vel[frame_idx])))

    def compute_batch(self, motion: StandardMotion) -> np.ndarray:
        return np.max(np.abs(motion.dof_vel), axis=-1).astype(np.float32)


class SmoothnessTerm(MetricTerm):
    """Mean per-joint angular velocity variance over the full clip.

    Corresponds directly to ``smoothness_penalty`` in the GMR benchmark::

        score = mean_j(var_t(Δq_j))

    This term is computed globally (not per-frame), so
    :meth:`compute_frame` returns the *global* smoothness value for every
    frame (useful for a constant HUD display).

    Lower is smoother.
    """

    name = "smoothness_penalty"
    weight = 0.1  # mirror GMR EvaluatorWeights default

    def compute_frame(
        self, motion: StandardMotion, frame_idx: int
    ) -> float:
        # Per-frame smoothness is ill-defined; return global value.
        return float(self._global_score(motion))

    def compute_batch(self, motion: StandardMotion) -> np.ndarray:
        score = self._global_score(motion)
        return np.full(motion.motion_length, score, dtype=np.float32)

    @staticmethod
    def _global_score(motion: StandardMotion) -> float:
        if motion.motion_length < 2:
            return 0.0
        vel = np.diff(motion.dof_pos, axis=0) * motion.fps  # (N-1, D)
        return float(np.mean(np.var(vel, axis=0)))


class RootLinVelTerm(MetricTerm):
    """Root linear velocity magnitude (m/s) per frame.

    Useful for detecting jumps / teleportation artefacts.
    """

    name = "root_lin_vel_magnitude"
    weight = 0.0  # informational only, not in composite by default

    def compute_frame(
        self, motion: StandardMotion, frame_idx: int
    ) -> float:
        return float(np.linalg.norm(motion.root_lin_vel[frame_idx]))

    def compute_batch(self, motion: StandardMotion) -> np.ndarray:
        return np.linalg.norm(motion.root_lin_vel, axis=-1).astype(np.float32)


class RootAngVelTerm(MetricTerm):
    """Root angular velocity magnitude (rad/s) per frame."""

    name = "root_ang_vel_magnitude"
    weight = 0.0

    def compute_frame(
        self, motion: StandardMotion, frame_idx: int
    ) -> float:
        return float(np.linalg.norm(motion.root_ang_vel[frame_idx]))

    def compute_batch(self, motion: StandardMotion) -> np.ndarray:
        return np.linalg.norm(motion.root_ang_vel, axis=-1).astype(np.float32)


class AMPFeatureStabilityTerm(MetricTerm):
    """AMP discriminator feature distribution stability.

    Measures the normalised standard deviation of the AMP-relevant feature
    vector across frames.  A low value indicates that the motion occupies
    a compact, consistent region in feature space (preferred for AMP
    training).

    The AMP feature vector is the concatenation of:
      ``[root_pos, root_rot, dof_pos, key_body_pos_local]`` (per frame)

    The metric is the mean of per-dimension ``std / (|mean| + ε)`` across
    all feature dimensions.
    """

    name = "amp_feature_stability"
    weight = 1.0

    def compute_frame(
        self, motion: StandardMotion, frame_idx: int
    ) -> float:
        # Return global stability (same for every frame, like smoothness).
        return float(self._global_score(motion))

    def compute_batch(self, motion: StandardMotion) -> np.ndarray:
        score = self._global_score(motion)
        return np.full(motion.motion_length, score, dtype=np.float32)

    @staticmethod
    def _global_score(motion: StandardMotion) -> float:
        feats = np.concatenate(
            [
                motion.root_pos,
                motion.root_rot,
                motion.dof_pos,
                motion.key_body_pos_local,
            ],
            axis=-1,
        )  # (N, D)
        std = np.std(feats, axis=0)
        mean_abs = np.abs(np.mean(feats, axis=0))
        eps = 1e-6
        cv = std / (mean_abs + eps)  # coefficient of variation per dim
        return float(np.mean(cv))


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------

class MetricsEngine:
    """Orchestrate quality metric evaluation over a ``StandardMotion`` clip.

    Parameters
    ----------
    terms:
        List of :class:`MetricTerm` instances to evaluate.  If ``None``,
        the default term set is used (see :meth:`_default_terms`).
    robot_model:
        Optional robot model passed to metric terms that require it
        (e.g. :class:`JointLimitViolationTerm`).
    """

    def __init__(
        self,
        terms: list[MetricTerm] | None = None,
        robot_model: RobotModel | None = None,
    ) -> None:
        if terms is None:
            terms = self._default_terms(robot_model)
        self._terms = list(terms)

    # ------------------------------------------------------------------
    # Term management
    # ------------------------------------------------------------------

    def register(self, term: MetricTerm) -> None:
        """Add a new :class:`MetricTerm` to the engine.

        Parameters
        ----------
        term:
            New metric term to register.
        """
        self._terms.append(term)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_frame(
        self, motion: StandardMotion, frame_idx: int
    ) -> dict[str, float]:
        """Compute all metrics for a single frame.

        Parameters
        ----------
        motion:
            Source motion clip.
        frame_idx:
            Frame index (0-based).

        Returns
        -------
        dict
            ``{metric_name: value, ..., "composite_score": float}``
        """
        result: dict[str, float] = {}
        composite = 0.0
        for term in self._terms:
            val = term.compute_frame(motion, frame_idx)
            result[term.name] = val
            if not np.isnan(val):
                composite += term.weight * val
        result["composite_score"] = composite
        return result

    def evaluate_batch(
        self, motion: StandardMotion
    ) -> dict[str, np.ndarray | float]:
        """Compute all metrics for every frame in the clip.

        Parameters
        ----------
        motion:
            Source motion clip.

        Returns
        -------
        dict
            ``{metric_name: np.ndarray shape (N_eff,), ...,
               "composite_score": np.ndarray, "composite_mean": float}``
        """
        result: dict[str, np.ndarray | float] = {}
        n = motion.motion_length
        composite = np.zeros(n, dtype=np.float32)

        for term in self._terms:
            arr = term.compute_batch(motion)
            result[term.name] = arr
            valid = ~np.isnan(arr)
            composite[valid] += term.weight * arr[valid]

        result["composite_score"] = composite
        result["composite_mean"] = float(np.nanmean(composite))
        return result

    def generate_report(
        self,
        motion: StandardMotion,
        output_path: str | Path,
    ) -> None:
        """Compute metrics and write a JSON report to *output_path*.

        Parameters
        ----------
        motion:
            Source motion clip.
        output_path:
            Destination JSON path.
        """
        report = self.evaluate_batch(motion)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialisable: dict = {
            "fps": motion.fps,
            "motion_length": motion.motion_length,
            "robot": motion.robot,
        }
        for k, v in report.items():
            if hasattr(v, "tolist"):
                serialisable[k] = v.tolist()
            else:
                serialisable[k] = v

        output_path.write_text(json.dumps(serialisable, indent=2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_terms(
        robot_model: RobotModel | None,
    ) -> list[MetricTerm]:
        """Return the default set of metric terms."""
        return [
            JointLimitViolationTerm(robot_model),
            JointVelSpikeTerm(),
            SmoothnessTerm(),
            RootLinVelTerm(),
            RootAngVelTerm(),
            AMPFeatureStabilityTerm(),
        ]
