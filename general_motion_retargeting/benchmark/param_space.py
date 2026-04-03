"""
IK Config Parameter Space for Optuna-based Auto-Tuning.

This module defines which parameters of an IK config JSON file are
optimisable, their allowed ranges, and how to map an Optuna trial's
suggested values back to a valid IK config dictionary.

Optimisation phases
-------------------
``'mvp'``
    Only the ``human_scale_table`` joint scale factors and the global IK
    solver ``damping`` are tuned (~15-16 parameters).  Fast to evaluate and
    a good first pass.

``'full'``
    Everything in *mvp* plus the per-task position / orientation weights in
    both ``ik_match_table1`` and ``ik_match_table2`` (~50-60 parameters in
    total for a typical config).
"""

from __future__ import annotations

import copy
import json
import pathlib
import warnings
from typing import Any

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
        ``'mvp'`` - only scale table + damping (fast, ~16 params).
        ``'full'`` - scale table + damping + task weights (~50-60 params).
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

    def suggest(self, trial: Any) -> dict[str, Any]:
        """Ask Optuna *trial* to suggest values for all parameters.

        Returns a flat ``{param_name: value}`` dict that can later be passed
        to :meth:`build_config`.  Note that ``'damping'`` is always included
        as a key, but it is a runtime argument (not part of the config JSON)
        and should be extracted before calling :meth:`build_config`.

        Parameters
        ----------
        trial : optuna.Trial
            An active Optuna trial object used for parameter suggestion.

        Returns
        -------
        dict
            Flat parameter dict with one key per optimisable parameter.
        """
        params: dict[str, Any] = {}

        # --- global solver damping (log-scale for wide range exploration) ---
        params["damping"] = trial.suggest_float("damping", DAMPING_LOW, DAMPING_HIGH, log=True)

        # --- optional: human height assumption in metres ---
        if self.tune_height:
            params["human_height_assumption"] = trial.suggest_float(
                "human_height_assumption",
                HEIGHT_LOW,
                HEIGHT_HIGH,
            )

        # --- optional: ground plane offset in metres ---
        if self.tune_ground:
            params["ground_height"] = trial.suggest_float("ground_height", -0.3, 0.3)

        # --- per-joint limb scale factors (uniform-range search) ---
        for joint_name in self.base_config.get("human_scale_table", {}):
            key = f"scale_{joint_name}"
            params[key] = trial.suggest_float(key, SCALE_LOW, SCALE_HIGH)

        # --- per-task weights (full mode only) ---
        # entry format: [body_name, pos_weight, rot_weight, pos_offset, rot_offset]
        if self.mode == "full":
            for table_key in ("ik_match_table1", "ik_match_table2"):
                table = self.base_config.get(table_key, {})
                for frame_name in table:
                    pos_key = f"{table_key}_{frame_name}_pos_weight"
                    rot_key = f"{table_key}_{frame_name}_rot_weight"
                    params[pos_key] = trial.suggest_float(pos_key, WEIGHT_LOW, WEIGHT_HIGH)
                    params[rot_key] = trial.suggest_float(rot_key, WEIGHT_LOW, WEIGHT_HIGH)

        return params

    def build_config(self, params: dict[str, Any]) -> dict:
        """Apply *params* (as returned by :meth:`suggest`) to the base config.

        Returns a deep copy of the base config with all recognised values
        substituted in.  The copy is safe to pass directly as
        ``ik_config_override`` to ``GeneralMotionRetargeting``.

        Extra keys that do not correspond to any config field (e.g.
        ``'damping'``, which is a runtime argument and *not* stored in the
        config JSON) are silently ignored.  A :class:`UserWarning` is emitted
        only for keys that look like mistakes (i.e. start with ``'scale_'``
        or ``'ik_match_table'`` but don't match any known entry), so that
        callers who forget to ``pop("damping")`` before calling this method
        are not silently misled (Bug 6 fix).

        Parameters
        ----------
        params : dict
            Flat parameter dict produced by :meth:`suggest` (or any dict
            that follows the same naming convention).

        Returns
        -------
        dict
            Modified deep copy of ``base_config``.
        """
        cfg = copy.deepcopy(self.base_config)

        # --- known runtime-only keys that are intentionally not in the config ---
        # ('damping' is a runtime argument passed separately to the retargeter)
        _runtime_keys = frozenset({"damping"})

        # --- global scalar fields ---
        if "human_height_assumption" in params:
            cfg["human_height_assumption"] = float(params["human_height_assumption"])
        if "ground_height" in params:
            cfg["ground_height"] = float(params["ground_height"])

        # --- human_scale_table: scale factors for each joint ---
        known_scale_keys = {
            f"scale_{j}" for j in cfg.get("human_scale_table", {})
        }
        for joint_name in list(cfg.get("human_scale_table", {}).keys()):
            key = f"scale_{joint_name}"
            if key in params:
                cfg["human_scale_table"][joint_name] = float(params[key])

        # --- task weights in ik_match_table1 / ik_match_table2 ---
        known_weight_keys: set[str] = set()
        for table_key in ("ik_match_table1", "ik_match_table2"):
            table = cfg.get(table_key, {})
            for frame_name, entry in table.items():
                pos_key = f"{table_key}_{frame_name}_pos_weight"
                rot_key = f"{table_key}_{frame_name}_rot_weight"
                known_weight_keys.add(pos_key)
                known_weight_keys.add(rot_key)
                if pos_key in params:
                    entry[1] = float(params[pos_key])
                if rot_key in params:
                    entry[2] = float(params[rot_key])

        # Warn about suspicious unrecognised keys so accidental pass-through
        # of e.g. 'scale_UnknownJoint' is surfaced rather than silently dropped.
        all_known = (
            _runtime_keys
            | {"human_height_assumption", "ground_height"}
            | known_scale_keys
            | known_weight_keys
        )
        for k in params:
            if k not in all_known and (
                k.startswith("scale_") or k.startswith("ik_match_table")
            ):
                warnings.warn(
                    f"build_config: unrecognised parameter key {k!r} - ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        return cfg

    def save_config(self, params: dict[str, Any], output_path: str | pathlib.Path) -> None:
        """Build the config from *params* and save it to *output_path* as JSON.

        Parameters
        ----------
        params : dict
            Flat parameter dict produced by :meth:`suggest`.
        output_path : str or pathlib.Path
            Destination path for the optimised JSON config file.
            Parent directories are created automatically if missing.
        """
        cfg = self.build_config(params)
        output_path = pathlib.Path(output_path)
        # Ensure the output directory exists before writing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(cfg, fh, indent=4)

    def n_params(self) -> int:
        """Return the total number of optimisable parameters.

        Returns
        -------
        int
            Count of parameters: 1 (damping) + optional height/ground +
            scale table size + 2 x task entries (full mode only).
        """
        # Start with 1 for the mandatory damping parameter
        n = 1
        if self.tune_height:
            n += 1
        if self.tune_ground:
            n += 1
        # One scale factor per joint in the human_scale_table
        n += len(self.base_config.get("human_scale_table", {}))
        if self.mode == "full":
            # Two weights (pos + rot) per entry in each IK match table
            for table_key in ("ik_match_table1", "ik_match_table2"):
                n += 2 * len(self.base_config.get(table_key, {}))
        return n

    def param_names(self) -> list[str]:
        """Return all parameter names in the same order as :meth:`suggest`.

        Returns
        -------
        list of str
            Parameter names, mirroring the insertion order of
            :meth:`suggest` so the two methods stay in sync.
        """
        names = ["damping"]
        if self.tune_height:
            names.append("human_height_assumption")
        if self.tune_ground:
            names.append("ground_height")
        # Scale factors in human_scale_table order
        for joint_name in self.base_config.get("human_scale_table", {}):
            names.append(f"scale_{joint_name}")
        if self.mode == "full":
            # Pos and rot weight for each frame entry in each table
            for table_key in ("ik_match_table1", "ik_match_table2"):
                for frame_name in self.base_config.get(table_key, {}):
                    names.append(f"{table_key}_{frame_name}_pos_weight")
                    names.append(f"{table_key}_{frame_name}_rot_weight")
        return names
