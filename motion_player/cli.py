"""Command-line interface for motion_player.

Entry point: ``motion-player``

Available sub-commands::

    motion-player play      <file>   --mapping <yaml>  [--backend mujoco|nv]
    motion-player evaluate  <file>   --mapping <yaml>  [--output report.json]
    motion-player audit     <file>   --robot-xml <xml>
    motion-player gen-sidecar <file> --robot-xml <xml> [--output meta.yaml]
    motion-player convert-nv <file>  [--output out.npy] [--wxyz]

All heavy dependencies are imported lazily inside each sub-command handler.
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_play(args: argparse.Namespace) -> int:
    """Launch the interactive motion player."""
    from motion_player import play_motion

    play_motion(
        motion_path=args.file,
        mapping_config=args.mapping,
        backend=args.backend,
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compute quality metrics and optionally save a report."""
    from motion_player import evaluate_motion

    report = evaluate_motion(
        motion_path=args.file,
        mapping_config=args.mapping,
        output_path=args.output,
    )
    composite = report.get("composite_mean", float("nan"))
    print(f"Composite quality score (mean): {composite:.4f}")
    if args.output:
        print(f"Full report written to: {args.output}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit DOF order between a motion file and a robot MJCF."""
    from motion_player.core.adapters import DatasetAdapter, ModelAdapter, DOFAuditor

    motion = DatasetAdapter().load(args.file)
    robot_model = ModelAdapter().load_mjcf(args.robot_xml)
    report = DOFAuditor().audit(motion, robot_model)

    print("=== DOF Audit Report ===")
    print(f"Order compatible: {report.is_order_compatible}")
    print(f"Matched ({len(report.matched)}): {report.matched[:5]}{'...' if len(report.matched) > 5 else ''}")
    if report.mismatched:
        print(f"Mismatched ({len(report.mismatched)}): {report.mismatched}")
    if report.unmatched_in_data:
        print(f"In data only: {report.unmatched_in_data}")
    if report.unmatched_in_model:
        print(f"In model only: {report.unmatched_in_model}")
    return 0 if report.is_order_compatible else 1


def cmd_gen_sidecar(args: argparse.Namespace) -> int:
    """Generate a sidecar YAML for a motion file."""
    import yaml
    from pathlib import Path
    from motion_player.core.adapters import DOFAuditor

    sidecar = DOFAuditor().generate_sidecar(
        source_pkl_path=args.file,
        robot_xml_path=args.robot_xml,
        robot_name=getattr(args, "robot_name", ""),
        source_pipeline=getattr(args, "source_pipeline", ""),
        gmr_ik_config=getattr(args, "gmr_ik_config", ""),
    )

    output = getattr(args, "output", None) or (
        str(Path(args.file).with_name(Path(args.file).stem + "_meta.yaml"))
    )
    with open(output, "w") as fh:
        yaml.dump(sidecar, fh, default_flow_style=False, allow_unicode=True)
    print(f"Sidecar written to: {output}")
    return 0


def cmd_convert_nv(args: argparse.Namespace) -> int:
    """Convert a standard motion file to NV AMP .npy format."""
    from pathlib import Path
    from motion_player.core.adapters import DatasetAdapter
    from motion_player.backends.nv_backend import save_nv_motion

    motion = DatasetAdapter().load(args.file)
    output = getattr(args, "output", None) or (
        str(Path(args.file).with_suffix("")) + "_nv.npy"
    )
    save_nv_motion(motion, output, wxyz=getattr(args, "wxyz", False))
    print(f"NV motion saved to: {output}")
    return 0


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="motion-player",
        description=(
            "Motion Dataset Player & Editor for humanoid robots. "
            "Use a sub-command to launch the viewer, evaluate quality, "
            "audit DOF order, or convert formats."
        ),
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ---- play ----
    p_play = sub.add_parser("play", help="Launch interactive motion player")
    p_play.add_argument("file", help="Path to *_standard.pkl or .npy file")
    p_play.add_argument("--mapping", default=None, metavar="YAML",
                        help="Path to mapping.yaml")
    p_play.add_argument("--backend", default="mujoco",
                        choices=["mujoco", "nv"],
                        help="Rendering backend (default: mujoco)")
    p_play.set_defaults(func=cmd_play)

    # ---- evaluate ----
    p_eval = sub.add_parser("evaluate", help="Compute quality metrics (no GUI)")
    p_eval.add_argument("file", help="Path to *_standard.pkl or .npy file")
    p_eval.add_argument("--mapping", default=None, metavar="YAML")
    p_eval.add_argument("--output", default=None, metavar="JSON",
                        help="Write report to this JSON file")
    p_eval.set_defaults(func=cmd_evaluate)

    # ---- audit ----
    p_audit = sub.add_parser("audit", help="Audit DOF order vs. robot MJCF")
    p_audit.add_argument("file", help="Path to *_standard.pkl or .npy file")
    p_audit.add_argument("--robot-xml", required=True, metavar="XML",
                         help="Path to the robot MJCF file")
    p_audit.set_defaults(func=cmd_audit)

    # ---- gen-sidecar ----
    p_sidecar = sub.add_parser(
        "gen-sidecar", help="Generate DOF-name sidecar YAML"
    )
    p_sidecar.add_argument("file", help="Path to *_standard.pkl or .npy file")
    p_sidecar.add_argument("--robot-xml", required=True, metavar="XML")
    p_sidecar.add_argument("--output", default=None, metavar="YAML",
                           help="Output sidecar path (default: <file>_meta.yaml)")
    p_sidecar.set_defaults(func=cmd_gen_sidecar)

    # ---- convert-nv ----
    p_nv = sub.add_parser(
        "convert-nv", help="Convert to NVIDIA AMP .npy format"
    )
    p_nv.add_argument("file", help="Path to *_standard.pkl or .npy file")
    p_nv.add_argument("--output", default=None, metavar="NPY")
    p_nv.add_argument("--wxyz", action="store_true",
                      help="Output quaternions in wxyz (scalar-first) order")
    p_nv.set_defaults(func=cmd_convert_nv)

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
