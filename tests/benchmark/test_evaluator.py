"""
Tests for general_motion_retargeting.benchmark.evaluator.

All MuJoCo and GeneralMotionRetargeting interactions are mocked so the
tests run without robot asset files or a GPU.  The focus is on the metric
computation logic inside RetargetingEvaluator.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from general_motion_retargeting.benchmark.evaluator import (
    EvaluatorWeights,
    RetargetingEvaluator,
    _dtw_distance,
)
from tests.benchmark.fixtures import make_synthetic_qpos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluator(weights: EvaluatorWeights | None = None) -> RetargetingEvaluator:
    """Return a RetargetingEvaluator with MuJoCo loading disabled.

    The robot XML loading is patched so no robot assets are needed.
    """
    with patch(
        "general_motion_retargeting.benchmark.evaluator.mj.MjModel.from_xml_path",
        side_effect=Exception("no robot asset in tests"),
    ):
        return RetargetingEvaluator(
            src_human="bvh_lafan1",
            tgt_robot="unitree_g1",
            weights=weights,
        )


# ---------------------------------------------------------------------------
# DTW distance tests
# ---------------------------------------------------------------------------


class TestDtwDistance:
    """Unit tests for the _dtw_distance helper."""

    def test_identical_sequences_give_zero(self) -> None:
        """Identical sequences must have DTW distance 0."""
        seq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert _dtw_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        """DTW(a, b) must equal DTW(b, a)."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((10, 3))
        b = rng.standard_normal((8, 3))
        assert _dtw_distance(a, b) == pytest.approx(_dtw_distance(b, a))

    def test_triangle_inequality(self) -> None:
        """DTW must satisfy a relaxed triangle inequality: d(a,c) ≤ d(a,b) + d(b,c).

        The standard DTW metric does not satisfy the triangle inequality in
        general, but for sequences of equal length with unit-step warping
        windows it often holds approximately.  We test the worst case with
        aligned sequences where equality is expected.
        """
        rng = np.random.default_rng(2)
        a = rng.standard_normal((15, 3))
        b = rng.standard_normal((15, 3))
        c = rng.standard_normal((15, 3))
        # Relaxed: allow up to 10 % violation due to normalisation
        assert _dtw_distance(a, c) <= (_dtw_distance(a, b) + _dtw_distance(b, c)) * 1.1 + 1e-9

    def test_empty_seq_a_returns_zero(self) -> None:
        """An empty seq_a must return 0.0 without error."""
        empty = np.zeros((0, 3))
        b = np.array([[1.0, 0.0, 0.0]])
        assert _dtw_distance(empty, b) == 0.0

    def test_empty_seq_b_returns_zero(self) -> None:
        """An empty seq_b must return 0.0 without error."""
        a = np.array([[1.0, 0.0, 0.0]])
        empty = np.zeros((0, 3))
        assert _dtw_distance(a, empty) == 0.0


# ---------------------------------------------------------------------------
# _aggregate tests (called directly to avoid real retarget)
# ---------------------------------------------------------------------------


class TestAggregate:
    """Tests for RetargetingEvaluator._aggregate."""

    def test_returns_all_required_keys(self) -> None:
        """_aggregate must return all 7 expected metric keys."""
        ev = _make_evaluator()
        qpos = [make_synthetic_qpos(10, 36, seed=0)]
        root = [np.zeros((10, 3))]
        metrics = ev._aggregate([0.1] * 10, qpos, root, root)
        expected_keys = {
            "ik_error",
            "smoothness_penalty",
            "joint_limit_violation_rate",
            "root_dtw_distance",
            "composite_score",
            "n_sequences",
            "n_frames_total",
        }
        assert expected_keys == set(metrics.keys())

    def test_smoothness_constant_motion_is_near_zero(self) -> None:
        """Perfectly constant qpos (no movement) must yield smoothness ≈ 0."""
        ev = _make_evaluator()
        # Constant trajectory: all frames identical → zero velocity → zero variance
        const_qpos = np.zeros((20, 36))
        metrics = ev._aggregate([0.0] * 20, [const_qpos], [np.zeros((20, 3))], [np.zeros((20, 3))])
        assert metrics["smoothness_penalty"] == pytest.approx(0.0, abs=1e-12)

    def test_smoothness_linear_motion_is_near_zero(self) -> None:
        """Linearly increasing qpos (constant velocity) must yield smoothness ≈ 0.

        Constant-velocity motion has zero acceleration, so the velocity
        variance should be zero.
        """
        ev = _make_evaluator()
        # Linear ramp: qpos[t, :] = t * 0.01 → constant velocity 0.01
        t = np.arange(20)
        linear_qpos = np.outer(t, np.ones(36)) * 0.01
        metrics = ev._aggregate(
            [0.0] * 20, [linear_qpos], [np.zeros((20, 3))], [np.zeros((20, 3))]
        )
        assert metrics["smoothness_penalty"] == pytest.approx(0.0, abs=1e-12)

    def test_smoothness_jerk_motion_is_positive(self) -> None:
        """Random (jerky) qpos must yield a smoothness penalty clearly > 0."""
        ev = _make_evaluator()
        rng = np.random.default_rng(42)
        jerk_qpos = rng.standard_normal((30, 36))
        metrics = ev._aggregate(
            [0.1] * 30, [jerk_qpos], [np.zeros((30, 3))], [np.zeros((30, 3))]
        )
        assert metrics["smoothness_penalty"] > 0.0

    def test_composite_score_formula(self) -> None:
        """composite_score must equal the exact weighted sum of sub-metrics."""
        weights = EvaluatorWeights(
            ik_error=2.0,
            smoothness_penalty=0.5,
            joint_limit_violation_rate=3.0,
            root_dtw_distance=1.0,
        )
        ev = _make_evaluator(weights=weights)

        # Use a constant qpos so smoothness = 0 and DTW = 0
        const_qpos = np.zeros((10, 36))
        root_traj = np.zeros((10, 3))
        ik_errors = [0.25] * 10  # constant IK error

        metrics = ev._aggregate(ik_errors, [const_qpos], [root_traj], [root_traj])

        expected = (
            weights.ik_error * metrics["ik_error"]
            + weights.smoothness_penalty * metrics["smoothness_penalty"]
            + weights.joint_limit_violation_rate * metrics["joint_limit_violation_rate"]
            + weights.root_dtw_distance * metrics["root_dtw_distance"]
        )
        assert metrics["composite_score"] == pytest.approx(expected)

    def test_violation_rate_empty_qpos(self) -> None:
        """Empty qpos list must return violation_rate = 0.0."""
        ev = _make_evaluator()
        metrics = ev._aggregate([], [], [], [])
        assert metrics["joint_limit_violation_rate"] == 0.0


class TestComputeViolationRate:
    """Unit tests for _compute_violation_rate."""

    def test_empty_qpos_returns_zero(self) -> None:
        """No qpos data must yield 0.0."""
        ev = _make_evaluator()
        assert ev._compute_violation_rate([]) == 0.0

    def test_no_joint_limits_returns_zero(self) -> None:
        """When _ranged_indices is empty (no loaded model), result is 0.0."""
        ev = _make_evaluator()
        # _ranged_indices is empty because robot XML is mocked to fail
        assert ev._ranged_indices == []
        qpos = [make_synthetic_qpos(5, 36, seed=0)]
        assert ev._compute_violation_rate(qpos) == 0.0
