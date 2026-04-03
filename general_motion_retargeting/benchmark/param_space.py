"""
IK Config Parameter Space for Optuna-based Auto-Tuning.

This module defines which parameters of an IK config JSON file are
optimisable, their allowed ranges, and how to map an Optuna trial's
suggested values back to a valid IK config dictionary.

Optimisation phases
-------------------
``'mvp'``
    Only the ``human_scale_table`` joint scale factors and the global IK
    solver ``damping`` are tuned (~15–16 parameters).  Fast to evaluate and
    a good first pass.

``'full'``
    Everything in *mvp* plus the per-task position / orientation weights in
    both ``ik_match_table1`` and ``ik_match_table2`` (~50–60 parameters in
    total for a typical config).
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Allowed range for joint scale factors in human_scale_table
SCALE_LOW = 0.3
SCALE_HIGH = 1.5

# Allowed range for IK solver damping
DAMPING_LOW = 1e-3
DAMPING_HIGH = 5.0

# Allowed range for human_height_assumption (metres)
HEIGHT_LOW = 1.4
HEIGHT_HIGH = 2.1

# Allowed range for task weights
WEIGHT_LOW = 0.0
WEIGHT_HIGH = 200.0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_config(path: str | pathlib.Path) -> dict:
    """Load a JSON IK config from *path* and return it as a plain dict."""
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# IKConfigParamSpace
# ---------------------------------------------------------------------------

class IKConfigParamSpace:
    """Defines the optimisable parameter space for a single IK config file.

    Parameters
    ----------
    base_config_path:
        Path to the *base* IK config JSON that will be used as a starting
        point.  All parameters not being tuned will keep their base values.
    mode:
        ``'mvp'`` – only scale table + damping (fast, ~16 params).
        ``'full'`` – scale table + damping + task weights (~50–60 params).
    tune_height:
        Whether to include ``human_height_assumption`` in the search space.
        Default ``True``.
    tune_ground:
        Whether to include ``ground_height`` in the search space.
        Default ``False`` (rarely needs changing).
    """

    def __init__(
        self,
        base_config_path: str | pathlib.Path,
        mode: str = "mvp",
        tune_height: bool = True,
        tune_ground: bool = False,
    ) -> None:
        if mode not in ("mvp", "full"):
            raise ValueError(f"mode must be 'mvp' or 'full', got {mode!r}")

        self.base_config_path = pathlib.Path(base_config_path)
        self.mode = mode
        self.tune_height = tune_height
        self.tune_ground = tune_ground
        self.base_config: dict = _load_config(self.base_config_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def suggest(self, trial: Any) -> Dict[str, Any]:
        """Ask Optuna *trial* to suggest values for all parameters.

        Returns a flat ``{param_name: value}`` dict that can later be passed
        to :meth:`build_config`.

        Parameters
        ----------
        trial:
            An ``optuna.Trial`` object.
        """
        params: Dict[str, Any] = {}

        # --- global solver damping ---
        params["damping"] = trial.suggest_float("damping", DAMPING_LOW, DAMPING_HIGH, log=True)

        # --- human_height_assumption ---
        if self.tune_height:
            params["human_height_assumption"] = trial.suggest_float(
                "human_height_assumption",
                HEIGHT_LOW,
                HEIGHT_HIGH,
            )

        # --- ground_height ---
        if self.tune_ground:
            params["ground_height"] = trial.suggest_float("ground_height", -0.3, 0.3)

        # --- human_scale_table ---
        for joint_name in self.base_config.get("human_scale_table", {}):
            key = f"scale_{joint_name}"
            params[key] = trial.suggest_float(key, SCALE_LOW, SCALE_HIGH)

        # --- task weights (full mode only) ---
        if self.mode == "full":
            for table_key in ("ik_match_table1", "ik_match_table2"):
                table = self.base_config.get(table_key, {})
                for frame_name, entry in table.items():
                    # entry format: [body_name, pos_weight, rot_weight, pos_offset, rot_offset]
                    pos_key = f"{table_key}_{frame_name}_pos_weight"
                    rot_key = f"{table_key}_{frame_name}_rot_weight"
                    params[pos_key] = trial.suggest_float(pos_key, WEIGHT_LOW, WEIGHT_HIGH)
                    params[rot_key] = trial.suggest_float(rot_key, 1.0, WEIGHT_HIGH)

        return params

    def build_config(self, params: Dict[str, Any]) -> dict:
        """Apply *params* (as returned by :meth:`suggest`) to the base config.

        Returns a deep copy of the base config with all suggested values
        substituted in.  The copy is safe to pass directly as
        ``ik_config_override`` to ``GeneralMotionRetargeting``.

        Parameters
        ----------
        params:
            Flat parameter dict produced by :meth:`suggest` (or any dict
            that follows the same naming convention).
        """
        cfg = copy.deepcopy(self.base_config)

        # --- global ---
        if "human_height_assumption" in params:
            cfg["human_height_assumption"] = float(params["human_height_assumption"])
        if "ground_height" in params:
            cfg["ground_height"] = float(params["ground_height"])

        # --- human_scale_table ---
        for joint_name in list(cfg.get("human_scale_table", {}).keys()):
            key = f"scale_{joint_name}"
            if key in params:
                cfg["human_scale_table"][joint_name] = float(params[key])

        # --- task weights (full mode) ---
        for table_key in ("ik_match_table1", "ik_match_table2"):
            table = cfg.get(table_key, {})
            for frame_name, entry in table.items():
                pos_key = f"{table_key}_{frame_name}_pos_weight"
                rot_key = f"{table_key}_{frame_name}_rot_weight"
                if pos_key in params:
                    entry[1] = float(params[pos_key])
                if rot_key in params:
                    entry[2] = float(params[rot_key])

        return cfg

    def save_config(self, params: Dict[str, Any], output_path: str | pathlib.Path) -> None:
        """Build the config from *params* and save it to *output_path* as JSON.

        Parameters
        ----------
        params:
            Flat parameter dict produced by :meth:`suggest`.
        output_path:
            Destination path for the optimised JSON config file.
        """
        cfg = self.build_config(params)
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(cfg, fh, indent=4)

    def n_params(self) -> int:
        """Return the approximate number of optimisable parameters."""
        n = 1  # damping
        if self.tune_height:
            n += 1
        if self.tune_ground:
            n += 1
        n += len(self.base_config.get("human_scale_table", {}))
        if self.mode == "full":
            for table_key in ("ik_match_table1", "ik_match_table2"):
                n += 2 * len(self.base_config.get(table_key, {}))
        return n

    def param_names(self) -> list[str]:
        """Return all parameter names in the same order as :meth:`suggest`."""
        names = ["damping"]
        if self.tune_height:
            names.append("human_height_assumption")
        if self.tune_ground:
            names.append("ground_height")
        for joint_name in self.base_config.get("human_scale_table", {}):
            names.append(f"scale_{joint_name}")
        if self.mode == "full":
            for table_key in ("ik_match_table1", "ik_match_table2"):
                for frame_name in self.base_config.get(table_key, {}):
                    names.append(f"{table_key}_{frame_name}_pos_weight")
                    names.append(f"{table_key}_{frame_name}_rot_weight")
        return names
