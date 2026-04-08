"""NVIDIA HumanoidViewMotion backend for motion_player (optional).

This module provides a thin adapter that allows replaying a
:class:`~motion_player.core.models.StandardMotion` clip using the NVIDIA
ASE / CALM ``HumanoidViewMotion`` task as a rendering backend.

.. important::
    This backend is **optional** and replay-only.  Editing operations are
    not supported through this backend.  All interactive editing must be
    done via the MuJoCo backend first; the edited motion can then be
    exported and replayed here.

Installation
------------
Install the NV backend extras::

    pip install motion-player[nv]

You must also have a working IsaacGym or IsaacSim environment.  The
``isaacgym`` package is **not** distributed on PyPI and must be installed
from NVIDIA's developer portal.

Usage
-----
::

    from motion_player.backends.nv_backend import NVHumanoidRenderer
    from motion_player.core.adapters import DatasetAdapter

    motion = DatasetAdapter().load("clip.pkl")
    renderer = NVHumanoidRenderer(asset_root="/path/to/assets")
    renderer.play(motion)

Design notes
------------
The NV backend wraps the ``HumanoidViewMotion`` IsaacGym task.  It
converts ``StandardMotion`` frames into the motion-file format expected by
the task (``*.npy`` dict with ``fps``, ``rotation``, ``root_translation``,
``dof_positions`` keys — or the richer AMP format depending on the task
version).

Full implementation is deferred to V1.5.  This file provides:
  - A clear interface stub that documents what V1.5 must implement.
  - Guards that raise ``ImportError`` with helpful messages if IsaacGym is
    not installed.
  - A ``to_nv_motion_dict`` utility that converts ``StandardMotion`` to
    the NV AMP motion dict format (independent of IsaacGym, useful for
    format conversion scripts).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from motion_player.core.models import RobotModel, StandardMotion
    from motion_player.core.playback import PlaybackEngine


# ---------------------------------------------------------------------------
# Format conversion utility (no IsaacGym dependency)
# ---------------------------------------------------------------------------

def to_nv_motion_dict(
    motion: "StandardMotion",
    wxyz: bool = False,
) -> dict[str, Any]:
    """Convert a ``StandardMotion`` to an NV AMP motion dict.

    The output dict can be saved with ``np.save(..., allow_pickle=True)``
    and loaded by the ``HumanoidViewMotion`` task.

    The NV AMP motion format (as used in ASE / CALM / PHC) expects::

        {
            "fps": float,
            "root_translation": np.ndarray (N, 3),
            "rotation": np.ndarray (N, 4),        # wxyz or xyzw
            "dof_positions": np.ndarray (N, D),
            "dof_vels": np.ndarray (N, D),
        }

    Parameters
    ----------
    motion:
        Source motion clip.
    wxyz:
        If ``True``, output quaternions in wxyz (scalar-first) format as
        expected by some NV task versions.  If ``False`` (default), keep
        the ``StandardMotion`` xyzw convention.

    Returns
    -------
    dict
        NV-compatible motion dictionary.
    """
    root_rot = motion.root_rot.copy()
    if wxyz:
        # Convert xyzw → wxyz: [x, y, z, w] → [w, x, y, z]
        # i.e. roll the last axis by +1 position
        root_rot = np.concatenate(
            [root_rot[:, 3:4], root_rot[:, :3]], axis=-1
        )

    return {
        "fps": motion.fps,
        "root_translation": motion.root_pos.copy(),
        "rotation": root_rot,
        "dof_positions": motion.dof_pos.copy(),
        "dof_vels": motion.dof_vel.copy(),
    }


def save_nv_motion(
    motion: "StandardMotion",
    output_path: str | Path,
    wxyz: bool = False,
) -> None:
    """Save a ``StandardMotion`` as an NV AMP ``.npy`` motion file.

    Parameters
    ----------
    motion:
        Source motion clip.
    output_path:
        Destination ``.npy`` file path.
    wxyz:
        If ``True``, save quaternions in wxyz format (see :func:`to_nv_motion_dict`).
    """
    d = to_nv_motion_dict(motion, wxyz=wxyz)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), d)


# ---------------------------------------------------------------------------
# NVHumanoidRenderer (stub — requires IsaacGym)
# ---------------------------------------------------------------------------

class NVHumanoidRenderer:
    """NVIDIA HumanoidViewMotion replay backend.

    .. note::
        This class is a **placeholder stub**.  Full implementation is
        planned for V1.5 and requires a working IsaacGym environment.

    Parameters
    ----------
    asset_root:
        Root directory of the IsaacGym assets (robot URDF / MJCF files
        for IsaacGym's asset loader).
    task_config:
        Optional task configuration dict (maps to IsaacGym task cfg).
        If ``None``, sensible defaults are used.
    """

    def __init__(
        self,
        asset_root: str | Path | None = None,
        task_config: dict | None = None,
    ) -> None:
        self._asset_root = str(asset_root) if asset_root else None
        self._task_config = task_config or {}
        self._check_isaacgym()

    # ------------------------------------------------------------------
    # BaseRenderer-like interface (stub)
    # ------------------------------------------------------------------

    def load_model(self, robot_model: "RobotModel") -> None:
        """Load the robot model into the IsaacGym environment.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError(
            "NVHumanoidRenderer.load_model() is not yet implemented (V1.5)."
        )

    def update_state(self, qpos: np.ndarray | None) -> None:
        """Write *qpos* into the Isaac environment DOF state.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError(
            "NVHumanoidRenderer.update_state() is not yet implemented (V1.5)."
        )

    def render_frame(self) -> None:
        """Render one frame in the Isaac viewer.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError(
            "NVHumanoidRenderer.render_frame() is not yet implemented (V1.5)."
        )

    def overlay_text(self, lines: list[str]) -> None:
        """Overlay text on the Isaac viewer.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError(
            "NVHumanoidRenderer.overlay_text() is not yet implemented (V1.5)."
        )

    def close(self) -> None:
        """Close the Isaac environment.

        .. note:: Not yet implemented.
        """
        # Graceful no-op for now.
        pass

    def run(
        self,
        engine: "PlaybackEngine",
        metrics_engine: Any = None,
    ) -> None:
        """Run the full playback loop in IsaacGym.

        .. note:: Not yet implemented.
        """
        raise NotImplementedError(
            "NVHumanoidRenderer.run() is not yet implemented (V1.5). "
            "Use MuJoCoRenderer for interactive playback, or call "
            "save_nv_motion() to export for manual HumanoidViewMotion use."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_isaacgym() -> None:
        """Raise a helpful ImportError if IsaacGym is not installed."""
        try:
            import isaacgym  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The NVIDIA backend requires IsaacGym, which is not "
                "installed.  Install it from the NVIDIA developer portal "
                "and then run:\n"
                "    pip install 'motion-player[nv]'\n"
                "Alternatively, use the default MuJoCo backend:\n"
                "    from motion_player.backends.mujoco_backend import MuJoCoRenderer"
            ) from exc
