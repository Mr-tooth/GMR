"""
GMR IK Config Auto-Tuner
========================

Automatically optimises the IK configuration for a given
(human-motion-source, robot) pair using Optuna black-box optimisation.

Quick start
-----------
**Phase 1 (MVP) – scale table + damping only:**

.. code-block:: bash

    python scripts/optimize_ik_config.py \\
        --robot unitree_g1 \\
        --src_human bvh_lafan1 \\
        --data_path /data/lafan1/ \\
        --dataset_type lafan1 \\
        --n_trials 100 \\
        --phase mvp \\
        --output_dir outputs/optimize_g1_lafan1

**Phase 2 (full) – adds task-weight optimisation:**

.. code-block:: bash

    python scripts/optimize_ik_config.py \\
        --robot unitree_g1 \\
        --src_human bvh_lafan1 \\
        --data_path /data/lafan1/ \\
        --dataset_type lafan1 \\
        --n_trials 300 \\
        --phase full \\
        --sampler cmaes \\
        --output_dir outputs/optimize_g1_lafan1_full

**Multi-objective (NSGA-II), IK error vs smoothness Pareto front:**

.. code-block:: bash

    python scripts/optimize_ik_config.py \\
        --robot unitree_g1 \\
        --src_human bvh_lafan1 \\
        --data_path /data/lafan1/ \\
        --multi_objective \\
        --n_trials 200 \\
        --output_dir outputs/pareto_g1

**Distributed / resumable (SQLite backend):**

.. code-block:: bash

    python scripts/optimize_ik_config.py \\
        --storage sqlite:///gmr_study.db \\
        --study_name g1_lafan1_mvp \\
        ...

Outputs
-------
``<output_dir>/best_ik_config.json``
    The IK config JSON with the best single-objective parameters (or the
    best Pareto-front member with lowest composite score for multi-objective
    runs).

``<output_dir>/best_params.json``
    The flat Optuna parameter dict for the best trial.

``<output_dir>/all_trials.csv``
    CSV table of all trial results (trial number, param values, metrics).

``<output_dir>/optimization_history.png``  *(requires matplotlib)*
    Plot of the objective value over trials.

``<output_dir>/param_importances.png``  *(requires matplotlib)*
    Bar chart of parameter importances computed by Optuna.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
import traceback
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so that ``general_motion_retargeting``
# can be imported even when the package is not installed.
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler, RandomSampler
except ImportError as _err:
    print(
        "[ERROR] optuna is required for IK config optimisation.\n"
        "Install it with:  pip install optuna\n"
        f"Original error: {_err}"
    )
    sys.exit(1)

from general_motion_retargeting.benchmark import (
    DatasetLoader,
    IKConfigParamSpace,
    RetargetingEvaluator,
)
from general_motion_retargeting.benchmark.evaluator import EvaluatorWeights
from general_motion_retargeting.params import IK_CONFIG_DICT


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automatically optimise GMR IK configs using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    ds = p.add_argument_group("Dataset")
    ds.add_argument(
        "--dataset_type",
        choices=["lafan1", "amass"],
        default="lafan1",
        help="Type of input motion dataset.",
    )
    ds.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root directory containing the dataset files.",
    )
    ds.add_argument(
        "--smplx_body_model_path",
        type=str,
        default=None,
        help=(
            "Path to the SMPL-X body model directory "
            "(required only when --dataset_type amass)."
        ),
    )
    ds.add_argument(
        "--max_sequences",
        type=int,
        default=10,
        help="Maximum number of sequences to load for evaluation.",
    )
    ds.add_argument(
        "--max_frames_per_seq",
        type=int,
        default=200,
        help="Maximum number of frames per sequence (for speed).",
    )

    # Robot / source
    rr = p.add_argument_group("Retargeting")
    rr.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        help="Target robot identifier (must be in ROBOT_XML_DICT).",
    )
    rr.add_argument(
        "--src_human",
        type=str,
        default="bvh_lafan1",
        help="Human motion source identifier (must be in IK_CONFIG_DICT).",
    )

    # Optimisation
    opt = p.add_argument_group("Optimisation")
    opt.add_argument(
        "--phase",
        choices=["mvp", "full"],
        default="mvp",
        help=(
            "Optimisation phase. 'mvp': scale table + damping only. "
            "'full': adds task weights."
        ),
    )
    opt.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Total number of Optuna trials.",
    )
    opt.add_argument(
        "--n_random_startup",
        type=int,
        default=20,
        help=(
            "Number of random trials before switching to the main sampler "
            "(ignored for CMA-ES and NSGA-II)."
        ),
    )
    opt.add_argument(
        "--sampler",
        choices=["tpe", "cmaes", "random"],
        default="tpe",
        help="Optimisation algorithm (single-objective runs).",
    )
    opt.add_argument(
        "--multi_objective",
        action="store_true",
        default=False,
        help=(
            "Enable multi-objective optimisation (IK error vs smoothness). "
            "Uses NSGA-II sampler and outputs a Pareto front."
        ),
    )
    opt.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL for persistence / resuming "
            "(e.g. 'sqlite:///gmr_study.db')."
        ),
    )
    opt.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name (auto-generated if not provided).",
    )
    opt.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel workers.  Values > 1 require --storage to "
            "be set (SQLite supports limited concurrency; prefer PostgreSQL "
            "for large n_jobs)."
        ),
    )

    # Objective weights
    wt = p.add_argument_group("Objective weights")
    wt.add_argument("--w_ik", type=float, default=1.0, help="Weight for IK tracking error.")
    wt.add_argument("--w_smooth", type=float, default=0.1, help="Weight for smoothness penalty.")
    wt.add_argument(
        "--w_limit", type=float, default=5.0, help="Weight for joint-limit violation rate."
    )
    wt.add_argument(
        "--w_root", type=float, default=0.5, help="Weight for root trajectory DTW distance."
    )

    # Output
    out = p.add_argument_group("Output")
    out.add_argument(
        "--output_dir",
        type=str,
        default="outputs/optimize",
        help="Directory to save optimised configs and reports.",
    )
    out.add_argument(
        "--tune_height",
        action="store_true",
        default=True,
        help="Include human_height_assumption in the search space.",
    )
    out.add_argument(
        "--no_tune_height",
        dest="tune_height",
        action="store_false",
    )
    out.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-trial and per-sequence details.",
    )

    return p


# ---------------------------------------------------------------------------
# Trial objective
# ---------------------------------------------------------------------------

def make_objective(
    param_space: IKConfigParamSpace,
    evaluator: RetargetingEvaluator,
    sequences: list,
    multi_objective: bool = False,
    verbose: bool = False,
):
    """Return an Optuna objective callable for a single or multi-objective study."""

    def objective(trial):
        # 1. Sample parameters from the search space
        params = param_space.suggest(trial)

        # 2. Build the modified IK config dict
        damping = params.pop("damping")
        ik_config = param_space.build_config(params)

        try:
            metrics = evaluator.evaluate(
                sequences=sequences,
                ik_config_override=ik_config,
                damping=damping,
            )
        except Exception as exc:
            if verbose:
                traceback.print_exc()
            if multi_objective:
                return float("inf"), float("inf")
            return float("inf")

        # Store metrics as user attributes for later analysis
        for key, val in metrics.items():
            trial.set_user_attr(key, val)

        if multi_objective:
            return metrics["ik_error"], metrics["smoothness_penalty"]
        return metrics["composite_score"]

    return objective


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def save_best_config(
    best_params: dict,
    param_space: IKConfigParamSpace,
    output_dir: pathlib.Path,
    best_damping: float,
) -> None:
    """Save the best IK config JSON and flat params JSON."""
    params_copy = dict(best_params)
    params_copy["damping"] = best_damping
    cfg = param_space.build_config(params_copy)
    # damping is not part of the IK config JSON; it is a runtime argument
    (output_dir / "best_ik_config.json").write_text(
        json.dumps(cfg, indent=4)
    )
    (output_dir / "best_params.json").write_text(
        json.dumps({"damping": best_damping, **params_copy}, indent=4)
    )
    print(f"\n[✓] Best IK config saved to {output_dir / 'best_ik_config.json'}")
    print(f"[✓] Best params saved to    {output_dir / 'best_params.json'}")


def save_trials_csv(study: "optuna.Study", output_dir: pathlib.Path) -> None:
    """Write all trial results to a CSV file."""
    csv_path = output_dir / "all_trials.csv"
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return

    # Collect all possible attribute and param names
    param_names = sorted({k for t in trials for k in t.params})
    attr_names = sorted({k for t in trials for k in t.user_attrs})

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["trial_number", "value"] + param_names + attr_names
        writer.writerow(header)
        for t in trials:
            val = t.value if not isinstance(t.values, (list, tuple)) else str(t.values)
            row = (
                [t.number, val]
                + [t.params.get(k, "") for k in param_names]
                + [t.user_attrs.get(k, "") for k in attr_names]
            )
            writer.writerow(row)

    print(f"[✓] Trial CSV saved to {csv_path}")


def save_plots(study: "optuna.Study", output_dir: pathlib.Path, multi_objective: bool) -> None:
    """Save optimisation history and param importance plots (requires matplotlib)."""
    try:
        import matplotlib  # noqa: F401
        import optuna.visualization.matplotlib as vis
    except ImportError:
        print("[INFO] matplotlib not available – skipping plots.")
        return

    try:
        if not multi_objective:
            ax = vis.plot_optimization_history(study)
            ax.figure.savefig(str(output_dir / "optimization_history.png"), dpi=120)
            print(f"[✓] Optimisation history plot saved.")

            ax2 = vis.plot_param_importances(study)
            ax2.figure.savefig(str(output_dir / "param_importances.png"), dpi=120)
            print(f"[✓] Parameter importance plot saved.")
        else:
            ax = vis.plot_pareto_front(study, target_names=["IK error", "Smoothness"])
            ax.figure.savefig(str(output_dir / "pareto_front.png"), dpi=120)
            print(f"[✓] Pareto front plot saved.")
    except Exception as exc:
        print(f"[WARN] Could not save plots: {exc}")


def print_benchmark_table(metrics: dict) -> None:
    """Pretty-print the benchmark metrics for the best trial."""
    print("\n" + "=" * 60)
    print("  Benchmark Results (best trial)")
    print("=" * 60)
    rows = [
        ("IK tracking error (mean)", f"{metrics.get('ik_error', float('nan')):.6f}"),
        ("Smoothness penalty (var)", f"{metrics.get('smoothness_penalty', float('nan')):.6f}"),
        ("Joint-limit violation rate", f"{metrics.get('joint_limit_violation_rate', float('nan')):.4%}"),
        ("Root trajectory DTW dist.", f"{metrics.get('root_dtw_distance', float('nan')):.6f}"),
        ("Composite score", f"{metrics.get('composite_score', float('nan')):.6f}"),
        ("Sequences evaluated", str(metrics.get('n_sequences', 'N/A'))),
        ("Total frames evaluated", str(metrics.get('n_frames_total', 'N/A'))),
    ]
    for label, value in rows:
        print(f"  {label:<38s}: {value}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Validate (src_human, robot) combination
    # ------------------------------------------------------------------
    if args.src_human not in IK_CONFIG_DICT:
        print(f"[ERROR] src_human={args.src_human!r} not found in IK_CONFIG_DICT.")
        print(f"  Available sources: {list(IK_CONFIG_DICT.keys())}")
        sys.exit(1)
    if args.robot not in IK_CONFIG_DICT[args.src_human]:
        print(
            f"[ERROR] robot={args.robot!r} not available for "
            f"src_human={args.src_human!r}."
        )
        print(
            f"  Available robots for this source: "
            f"{list(IK_CONFIG_DICT[args.src_human].keys())}"
        )
        sys.exit(1)

    base_config_path = IK_CONFIG_DICT[args.src_human][args.robot]
    print(f"[GMR-OPT] Base IK config : {base_config_path}")
    print(f"[GMR-OPT] Robot          : {args.robot}")
    print(f"[GMR-OPT] Source human   : {args.src_human}")
    print(f"[GMR-OPT] Phase          : {args.phase}")
    print(f"[GMR-OPT] Trials         : {args.n_trials}")

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    print(f"\n[GMR-OPT] Loading dataset ({args.dataset_type}) from {args.data_path} …")
    loader = DatasetLoader(
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        max_sequences=args.max_sequences,
        max_frames_per_seq=args.max_frames_per_seq,
        smplx_body_model_path=args.smplx_body_model_path,
        verbose=True,
    )
    sequences = loader.load_sequences()
    if not sequences:
        print("[ERROR] No sequences loaded. Check --data_path.")
        sys.exit(1)
    total_frames = sum(len(f) for f, _ in sequences)
    print(
        f"[GMR-OPT] Loaded {len(sequences)} sequence(s), "
        f"{total_frames} frames total."
    )

    # ------------------------------------------------------------------
    # 3. Build param space and evaluator
    # ------------------------------------------------------------------
    param_space = IKConfigParamSpace(
        base_config_path=base_config_path,
        mode=args.phase,
        tune_height=args.tune_height,
    )
    print(
        f"[GMR-OPT] Optimising {param_space.n_params()} parameter(s): "
        f"{param_space.param_names()}"
    )

    weights = EvaluatorWeights(
        ik_error=args.w_ik,
        smoothness_penalty=args.w_smooth,
        joint_limit_violation_rate=args.w_limit,
        root_dtw_distance=args.w_root,
    )
    evaluator = RetargetingEvaluator(
        src_human=args.src_human,
        tgt_robot=args.robot,
        weights=weights,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # 4. Build Optuna study
    # ------------------------------------------------------------------
    study_name = args.study_name or (
        f"gmr_{args.src_human}_{args.robot}_{args.phase}_{int(time.time())}"
    )

    if args.multi_objective:
        sampler = NSGAIISampler()
        directions = ["minimize", "minimize"]
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            sampler=sampler,
            directions=directions,
            load_if_exists=True,
        )
    else:
        if args.sampler == "tpe":
            sampler = TPESampler(n_startup_trials=args.n_random_startup)
        elif args.sampler == "cmaes":
            sampler = CmaEsSampler(n_startup_trials=args.n_random_startup)
        else:
            sampler = RandomSampler()

        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            sampler=sampler,
            direction="minimize",
            load_if_exists=True,
        )

    # ------------------------------------------------------------------
    # 5. Run optimisation
    # ------------------------------------------------------------------
    objective_fn = make_objective(
        param_space=param_space,
        evaluator=evaluator,
        sequences=sequences,
        multi_objective=args.multi_objective,
        verbose=args.verbose,
    )

    completed_before = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_new_trials = max(0, args.n_trials - completed_before)

    print(
        f"\n[GMR-OPT] Starting optimisation: {n_new_trials} new trial(s) "
        f"({completed_before} already completed).\n"
    )

    optuna.logging.set_verbosity(
        optuna.logging.INFO if args.verbose else optuna.logging.WARNING
    )

    study.optimize(
        objective_fn,
        n_trials=n_new_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=not args.verbose,
    )

    # ------------------------------------------------------------------
    # 6. Extract best result and save outputs
    # ------------------------------------------------------------------
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print("[WARN] No trials completed successfully.")
        sys.exit(1)

    if args.multi_objective:
        # Pick the Pareto-front member with the lowest composite score
        # (sum of normalised objectives as a simple scalarisation)
        pareto_trials = study.best_trials
        best_trial = min(
            pareto_trials,
            key=lambda t: sum(v for v in t.values if v != float("inf")),
        )
        print(
            f"\n[GMR-OPT] Pareto front has {len(pareto_trials)} member(s). "
            f"Selected trial #{best_trial.number} (lowest sum of objectives)."
        )
    else:
        best_trial = study.best_trial
        print(
            f"\n[GMR-OPT] Best trial: #{best_trial.number} "
            f"with composite score = {best_trial.value:.6f}"
        )

    best_params = dict(best_trial.params)
    best_damping = best_params.pop("damping", 0.5)

    # Re-evaluate best config to get full metrics for the benchmark table
    best_ik_config = param_space.build_config({**best_params, "damping": best_damping})
    print("\n[GMR-OPT] Re-evaluating best config for benchmark report …")
    best_metrics = evaluator.evaluate(
        sequences=sequences,
        ik_config_override=best_ik_config,
        damping=best_damping,
    )
    print_benchmark_table(best_metrics)

    # Save outputs
    save_best_config(best_params, param_space, output_dir, best_damping)
    save_trials_csv(study, output_dir)
    save_plots(study, output_dir, args.multi_objective)

    # Save benchmark metrics JSON
    metrics_path = output_dir / "best_metrics.json"
    metrics_path.write_text(json.dumps(best_metrics, indent=4))
    print(f"[✓] Benchmark metrics saved to {metrics_path}")

    print(f"\n[GMR-OPT] Done. All outputs in: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
