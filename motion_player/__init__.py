"""motion_player — Motion Dataset Player & Editor for Humanoid Robots.

A standalone tool for playing back, inspecting, and editing standardized
robot motion datasets (from rsl-rl-ex AMP motion_loader format) using
MuJoCo as the primary rendering backend.

Basic usage::

    from motion_player import play_motion, evaluate_motion

    play_motion("clip_standard.pkl", mapping_config="mapping.yaml")

CLI usage::

    motion-player play clip_standard.pkl --mapping mapping.yaml

All heavy dependencies (mujoco, pinocchio, isaacgym) are imported lazily
inside their respective backend / solver modules, so this package is
safe to import in environments where not all backends are installed.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "play_motion",
    "evaluate_motion",
    "__version__",
]


def play_motion(
    motion_path: str,
    mapping_config: str | None = None,
    backend: str = "mujoco",
    **kwargs,
) -> None:
    """Launch the interactive motion player.

    Parameters
    ----------
    motion_path:
        Path to a ``*_standard.pkl`` or ``.npy`` standard motion file.
    mapping_config:
        Path to a ``mapping.yaml`` that specifies DOF order, joint name
        mapping, sign flips, and the robot MJCF path.  If ``None``, the
        player will attempt auto-detection (requires a sidecar YAML).
    backend:
        Rendering backend to use.  ``"mujoco"`` (default) or ``"nv"``.
    **kwargs:
        Additional keyword arguments forwarded to the backend.

    Raises
    ------
    ImportError
        If the requested backend is not installed.
    FileNotFoundError
        If *motion_path* or the MJCF referenced in *mapping_config* cannot
        be found.
    """
    # Lazy import to avoid pulling in heavy deps at package import time.
    from motion_player.core.adapters import DatasetAdapter, ModelAdapter
    from motion_player.core.playback import PlaybackEngine

    if backend == "mujoco":
        from motion_player.backends.mujoco_backend import MuJoCoRenderer as Renderer
    elif backend == "nv":
        from motion_player.backends.nv_backend import NVHumanoidRenderer as Renderer
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'mujoco' or 'nv'.")

    motion = DatasetAdapter().load(motion_path)
    mapping = ModelAdapter.load_mapping(mapping_config) if mapping_config else None
    robot_model = ModelAdapter().load_mjcf(
        mapping["robot_mjcf_path"], mapping_config=mapping
    ) if mapping else None

    renderer = Renderer(**kwargs)
    if robot_model is not None:
        renderer.load_model(robot_model)

    engine = PlaybackEngine(motion)
    engine.on_frame_change(
        lambda idx, m: renderer.update_state(
            ModelAdapter().motion_to_qpos(m, idx, robot_model) if robot_model else None
        )
    )

    try:
        renderer.run(engine)
    finally:
        renderer.close()


def evaluate_motion(
    motion_path: str,
    mapping_config: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Compute quality metrics for a standard motion file (no GUI).

    Parameters
    ----------
    motion_path:
        Path to a ``*_standard.pkl`` or ``.npy`` standard motion file.
    mapping_config:
        Path to ``mapping.yaml`` (needed for joint-limit checks).
    output_path:
        If provided, write the report to this path as JSON.

    Returns
    -------
    dict
        Metrics report dictionary.  Keys include per-frame arrays and
        aggregate scalars for each registered :class:`MetricTerm`.
    """
    from motion_player.core.adapters import DatasetAdapter, ModelAdapter
    from motion_player.core.metrics import MetricsEngine

    motion = DatasetAdapter().load(motion_path)
    mapping = ModelAdapter.load_mapping(mapping_config) if mapping_config else None
    robot_model = (
        ModelAdapter().load_mjcf(mapping["robot_mjcf_path"], mapping_config=mapping)
        if mapping
        else None
    )

    engine = MetricsEngine(robot_model=robot_model)
    report = engine.evaluate_batch(motion)

    if output_path is not None:
        import json
        from pathlib import Path
        # Convert numpy arrays to lists for JSON serialisation.
        serialisable = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in report.items()
        }
        Path(output_path).write_text(json.dumps(serialisable, indent=2))

    return report
