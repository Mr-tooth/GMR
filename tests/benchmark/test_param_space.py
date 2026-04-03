"""
Tests for general_motion_retargeting.benchmark.param_space.

All tests use a minimal IK config derived from bvh_lafan1_to_g1.json
and a stub Optuna trial so that no real Optuna study needs to be created.
No MuJoCo dependency is required.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from general_motion_retargeting.benchmark.param_space import IKConfigParamSpace
from tests.benchmark.fixtures import make_lafan1_ik_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param_space(mode: str = "mvp", tune_height: bool = True) -> IKConfigParamSpace:
    """Return an IKConfigParamSpace backed by the real bvh_lafan1_to_g1 config."""
    config_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "general_motion_retargeting"
        / "ik_configs"
        / "bvh_lafan1_to_g1.json"
    )
    return IKConfigParamSpace(config_path, mode=mode, tune_height=tune_height)


def _make_mock_trial(param_space: IKConfigParamSpace, value: float = 0.5) -> MagicMock:
    """Return a mock Optuna trial whose suggest_float always returns *value*."""
    trial = MagicMock()
    trial.suggest_float = MagicMock(return_value=value)
    return trial


# ---------------------------------------------------------------------------
# Parametrisation helpers
# ---------------------------------------------------------------------------


def _mvp_scale_table_size(ps: IKConfigParamSpace) -> int:
    """Return the number of joints in the human_scale_table."""
    return len(ps.base_config.get("human_scale_table", {}))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMvpParamCount:
    """n_params() must equal 1 (damping) + tune_height + scale_table_size."""

    def test_with_tune_height(self) -> None:
        ps = _make_param_space(mode="mvp", tune_height=True)
        expected = 1 + 1 + _mvp_scale_table_size(ps)
        assert ps.n_params() == expected

    def test_without_tune_height(self) -> None:
        ps = _make_param_space(mode="mvp", tune_height=False)
        expected = 1 + 0 + _mvp_scale_table_size(ps)
        assert ps.n_params() == expected


class TestFullParamCount:
    """Full mode must include all task-weight parameters from both IK tables."""

    def test_full_is_larger_than_mvp(self) -> None:
        ps_mvp = _make_param_space(mode="mvp")
        ps_full = _make_param_space(mode="full")
        # Full mode adds 2 weights per frame per table
        assert ps_full.n_params() > ps_mvp.n_params()

    def test_full_param_count_formula(self) -> None:
        ps = _make_param_space(mode="full")
        cfg = ps.base_config
        # Base: damping + tune_height + scale_table
        base = 1 + 1 + _mvp_scale_table_size(ps)
        # Plus 2 weights per entry in each IK table
        extra = sum(
            2 * len(cfg.get(k, {}))
            for k in ("ik_match_table1", "ik_match_table2")
        )
        assert ps.n_params() == base + extra


class TestSuggestReturnsAllKeys:
    """suggest(trial) must return exactly the keys listed by param_names()."""

    def test_mvp_keys_match(self) -> None:
        ps = _make_param_space(mode="mvp")
        trial = _make_mock_trial(ps)
        params = ps.suggest(trial)
        assert set(params.keys()) == set(ps.param_names())

    def test_full_keys_match(self) -> None:
        ps = _make_param_space(mode="full")
        trial = _make_mock_trial(ps)
        params = ps.suggest(trial)
        assert set(params.keys()) == set(ps.param_names())


class TestBuildConfigRoundtrip:
    """build_config(suggest(trial)) must produce a valid config dict."""

    def test_output_is_dict(self) -> None:
        ps = _make_param_space()
        trial = _make_mock_trial(ps)
        params = ps.suggest(trial)
        cfg = ps.build_config(params)
        assert isinstance(cfg, dict)

    def test_scale_table_values_are_floats(self) -> None:
        ps = _make_param_space()
        trial = _make_mock_trial(ps, value=0.7)
        params = ps.suggest(trial)
        cfg = ps.build_config(params)
        for jname, val in cfg["human_scale_table"].items():
            assert isinstance(val, float), f"scale for {jname!r} is not float"

    def test_height_assumption_updated(self) -> None:
        ps = _make_param_space(tune_height=True)
        trial = _make_mock_trial(ps, value=1.75)
        params = ps.suggest(trial)
        cfg = ps.build_config(params)
        assert cfg["human_height_assumption"] == pytest.approx(1.75)

    def test_full_mode_weights_updated(self) -> None:
        ps = _make_param_space(mode="full")
        trial = _make_mock_trial(ps, value=42.0)
        params = ps.suggest(trial)
        cfg = ps.build_config(params)
        # Every pos and rot weight in ik_match_table1 should be 42.0
        for entry in cfg["ik_match_table1"].values():
            assert entry[1] == pytest.approx(42.0)  # pos_weight index
            assert entry[2] == pytest.approx(42.0)  # rot_weight index


class TestBuildConfigIndependence:
    """Two successive build_config() calls must return independent deep copies."""

    def test_modify_first_does_not_affect_second(self) -> None:
        ps = _make_param_space()
        trial = _make_mock_trial(ps)
        params = ps.suggest(trial)

        cfg_a = ps.build_config(params)
        cfg_b = ps.build_config(params)

        # Mutate cfg_a's scale table
        first_joint = next(iter(cfg_a["human_scale_table"]))
        cfg_a["human_scale_table"][first_joint] = 99.0

        # cfg_b must be unaffected
        assert cfg_b["human_scale_table"][first_joint] != 99.0


class TestSaveConfigValidJson:
    """save_config must write a valid JSON file with the same top-level keys."""

    def test_roundtrip_json(self) -> None:
        ps = _make_param_space()
        trial = _make_mock_trial(ps)
        params = ps.suggest(trial)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = pathlib.Path(tmpdir) / "sub" / "best.json"
            ps.save_config(params, out_path)

            assert out_path.exists(), "Output file was not created"
            with open(out_path) as fh:
                loaded = json.load(fh)

        # Must contain the same structural keys as the original config
        original_keys = set(ps.base_config.keys())
        assert original_keys.issubset(set(loaded.keys()))


class TestInvalidModeRaises:
    """IKConfigParamSpace must raise ValueError for unknown modes."""

    def test_invalid_mode(self) -> None:
        config_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "general_motion_retargeting"
            / "ik_configs"
            / "bvh_lafan1_to_g1.json"
        )
        with pytest.raises(ValueError, match="mode must be"):
            IKConfigParamSpace(config_path, mode="bad_mode")
