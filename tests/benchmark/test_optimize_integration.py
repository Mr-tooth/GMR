"""
Integration tests for scripts/optimize_ik_config.py.

All retargeting and evaluation is mocked so that Optuna trials complete
in milliseconds without real motion data.  The tests verify:

1. The objective closure returns a finite float (single-objective).
2. A single-trial Optuna study runs end-to-end and writes output files.
3. The multi-objective (NSGA-II) mode returns two finite values per trial.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest

from general_motion_retargeting.benchmark.param_space import IKConfigParamSpace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "optimize_ik_config.py"
)

_IK_CONFIG_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "general_motion_retargeting"
    / "ik_configs"
    / "bvh_lafan1_to_g1.json"
)


def _load_script_module() -> ModuleType:
    """Dynamically import optimize_ik_config.py as a module.

    This avoids relying on it being installed as a console-script entry point.
    """
    spec = importlib.util.spec_from_file_location("optimize_ik_config", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_param_space(mode: str = "mvp") -> IKConfigParamSpace:
    """Return a real IKConfigParamSpace backed by the G1 config."""
    return IKConfigParamSpace(_IK_CONFIG_PATH, mode=mode, tune_height=True)


def _fake_metrics(**kwargs: Any) -> dict:
    """Return a plausible metrics dict for a mocked evaluate() call."""
    return {
        "ik_error": 0.05,
        "smoothness_penalty": 0.002,
        "joint_limit_violation_rate": 0.01,
        "root_dtw_distance": 0.03,
        "composite_score": 0.15,
        "n_sequences": 1,
        "n_frames_total": 10,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMakeObjectiveReturnFinite:
    """The objective closure must return a finite value when evaluate succeeds."""

    def test_single_objective_returns_float(self) -> None:
        mod = _load_script_module()

        ps = _make_param_space()
        # Mock the evaluator's evaluate method to return fixed metrics
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = _fake_metrics()

        sequences = [([{"Hips": (np.zeros(3), np.array([1, 0, 0, 0]))}], 1.75)]

        objective_fn = mod.make_objective(
            param_space=ps,
            evaluator=mock_evaluator,
            sequences=sequences,
            multi_objective=False,
            verbose=False,
        )

        # Create a real Optuna trial via a minimalistic study
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())

        results: list[float] = []

        def _capture(trial: optuna.Trial) -> float:
            value = objective_fn(trial)
            results.append(value)
            return value

        study.optimize(_capture, n_trials=1, catch=())
        assert len(results) == 1
        assert np.isfinite(results[0])

    def test_multi_objective_returns_two_finite_values(self) -> None:
        mod = _load_script_module()

        ps = _make_param_space()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = _fake_metrics()

        sequences = [([{"Hips": (np.zeros(3), np.array([1, 0, 0, 0]))}], 1.75)]

        objective_fn = mod.make_objective(
            param_space=ps,
            evaluator=mock_evaluator,
            sequences=sequences,
            multi_objective=True,
            verbose=False,
        )

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=0),
        )

        results: list[tuple[float, float]] = []

        def _capture(trial: optuna.Trial) -> tuple[float, float]:
            values = objective_fn(trial)
            results.append(values)
            return values

        study.optimize(_capture, n_trials=1, catch=())
        assert len(results) == 1
        v0, v1 = results[0]
        assert np.isfinite(v0) and np.isfinite(v1)


class TestStudySingleTrial:
    """A single-trial study must complete and write best_ik_config.json."""

    def test_single_trial_writes_output(self) -> None:
        mod = _load_script_module()

        ps = _make_param_space()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = _fake_metrics()

        sequences = [([{"Hips": (np.zeros(3), np.array([1, 0, 0, 0]))}], 1.75)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.RandomSampler(),
            )
            objective_fn = mod.make_objective(
                param_space=ps,
                evaluator=mock_evaluator,
                sequences=sequences,
                multi_objective=False,
                verbose=False,
            )
            study.optimize(objective_fn, n_trials=1, catch=())

            # Reproduce what main() does after optimisation
            best_params = dict(study.best_params)
            best_damping = best_params.pop("damping", 0.5)
            mod.save_best_config(best_params, ps, output_dir, best_damping)
            mod.save_trials_csv(study, output_dir)

            config_path = output_dir / "best_ik_config.json"
            assert config_path.exists(), "best_ik_config.json was not created"

            # Verify the output is valid JSON with required fields
            with open(config_path) as fh:
                cfg = json.load(fh)
            assert "human_scale_table" in cfg
            assert "ik_match_table1" in cfg
