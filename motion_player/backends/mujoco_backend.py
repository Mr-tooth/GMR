"""MuJoCo kinematic replay backend for motion_player.

This backend renders a :class:`~motion_player.core.models.StandardMotion`
clip by writing joint positions directly into ``mjData.qpos`` and calling
``mj_forward`` each frame.  No physics simulation is involved; this is a
pure *kinematic replay* and is the recommended default for data inspection.

The backend wraps ``mujoco.viewer.launch_passive`` (MuJoCo ≥ 3.0) and
registers keyboard callbacks for playback control.

Dependencies
------------
This module imports ``mujoco`` at class instantiation time.  The import is
**not** at module level so that the rest of the ``motion_player`` package
remains importable without MuJoCo installed.

Usage
-----
::

    from motion_player.backends.mujoco_backend import MuJoCoRenderer
    from motion_player.core.playback import PlaybackEngine
    from motion_player.core.adapters import DatasetAdapter, ModelAdapter

    motion  = DatasetAdapter().load("clip.pkl")
    mapping = ModelAdapter.load_mapping("mapping.yaml")
    robot   = ModelAdapter().load_mjcf(mapping["robot_mjcf_path"], mapping)

    renderer = MuJoCoRenderer(mapping["robot_mjcf_path"])
    renderer.load_model(robot)

    engine = PlaybackEngine(motion)
    engine.on_frame_change(
        lambda idx, m: renderer.update_state(
            ModelAdapter().motion_to_qpos(m, idx, robot)
        )
    )
    renderer.run(engine)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from motion_player.core.models import RobotModel
    from motion_player.core.playback import PlaybackEngine


class MuJoCoRenderer:
    """MuJoCo kinematic replay renderer.

    Parameters
    ----------
    xml_path:
        Path to the robot MJCF file.
    camera_mode:
        ``"follow_root"`` — camera tracks the floating base (default).
        ``"fixed"``       — static camera position.
        ``"free"``        — user-controlled free camera.
    show_hud:
        If ``True`` (default), overlay quality metric text on the viewer.
    headless:
        If ``True``, run in offscreen mode (no window, for testing / batch
        rendering).  Default ``False``.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        camera_mode: str = "follow_root",
        show_hud: bool = True,
        headless: bool = False,
    ) -> None:
        # Defer heavy imports to avoid penalising package-level import time.
        import mujoco as mj  # noqa: PLC0415

        self._mj = mj
        self._xml_path = str(xml_path) if xml_path else None
        self._camera_mode = camera_mode
        self._show_hud = show_hud
        self._headless = headless

        self._model: "mj.MjModel | None" = None
        self._data: "mj.MjData | None" = None
        self._viewer = None
        self._hud_lines: list[str] = []
        self._robot_model: RobotModel | None = None

    # ------------------------------------------------------------------
    # BaseRenderer interface
    # ------------------------------------------------------------------

    def load_model(self, robot_model: RobotModel) -> None:
        """Load the robot MJCF into MuJoCo.

        Parameters
        ----------
        robot_model:
            Populated :class:`~motion_player.core.models.RobotModel`.
        """
        mj = self._mj
        xml = robot_model.mjcf_path or self._xml_path
        if xml is None:
            raise ValueError(
                "No MJCF path provided.  Pass xml_path to MuJoCoRenderer "
                "or ensure RobotModel.mjcf_path is set."
            )
        self._model = mj.MjModel.from_xml_path(str(xml))
        self._data = mj.MjData(self._model)
        self._robot_model = robot_model

    def update_state(self, qpos: "np.ndarray | None") -> None:  # type: ignore[name-defined]
        """Write *qpos* into ``mjData`` and advance kinematics.

        Parameters
        ----------
        qpos:
            Full ``qpos`` array as produced by
            :meth:`~motion_player.core.adapters.ModelAdapter.motion_to_qpos`.
            If ``None``, the call is a no-op.
        """
        if qpos is None or self._model is None or self._data is None:
            return
        import numpy as np  # noqa: PLC0415

        n = min(len(qpos), self._data.qpos.shape[0])
        self._data.qpos[:n] = qpos[:n]
        self._mj.mj_forward(self._model, self._data)

    def render_frame(self) -> None:
        """Sync the viewer to display the current ``mjData`` state.

        This is a no-op if the viewer has not been launched yet.
        """
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def overlay_text(self, lines: list[str]) -> None:
        """Update the HUD text overlay.

        Parameters
        ----------
        lines:
            List of text lines to display in the viewer's top-left corner.
        """
        self._hud_lines = list(lines)

    def close(self) -> None:
        """Close the viewer and release MuJoCo resources."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001
                pass
            self._viewer = None

    # ------------------------------------------------------------------
    # High-level run loop
    # ------------------------------------------------------------------

    def run(
        self,
        engine: PlaybackEngine,
        metrics_engine: "motion_player.core.metrics.MetricsEngine | None" = None,
    ) -> None:
        """Start the interactive viewer loop.

        This method blocks until the viewer window is closed by the user.
        Keyboard shortcuts are registered with the viewer; see
        ``docs/design.md §5.2`` for the full keybinding table.

        Parameters
        ----------
        engine:
            The :class:`~motion_player.core.playback.PlaybackEngine` to
            drive.
        metrics_engine:
            Optional :class:`~motion_player.core.metrics.MetricsEngine`
            for real-time HUD display.
        """
        if self._model is None or self._data is None:
            raise RuntimeError(
                "Call load_model() before run()."
            )

        mj = self._mj

        if self._headless:
            self._run_headless(engine, metrics_engine)
            return

        with mj.viewer.launch_passive(
            self._model,
            self._data,
            key_callback=lambda key: self._key_callback(key, engine),
        ) as viewer:
            self._viewer = viewer
            self._setup_camera(viewer)

            while viewer.is_running():
                engine.tick()

                if self._show_hud and metrics_engine is not None:
                    frame_metrics = metrics_engine.evaluate_frame(
                        engine.current_motion, engine.current_frame
                    )
                    hud = [
                        f"Frame: {engine.current_frame}/{engine.current_motion.motion_length - 1}",
                        f"Speed: {engine.speed:.2f}x",
                        f"Playing: {'yes' if engine.is_playing else 'no'}",
                        "---",
                    ]
                    for k, v in frame_metrics.items():
                        if k != "composite_score":
                            hud.append(f"{k}: {v:.4f}")
                    hud.append(f"composite: {frame_metrics.get('composite_score', 0.0):.4f}")
                    self.overlay_text(hud)

                viewer.sync()
                # Yield CPU time when paused to avoid busy-waiting.
                if not engine.is_playing:
                    time.sleep(0.016)

    # ------------------------------------------------------------------
    # Ghost overlay (A/B comparison)
    # ------------------------------------------------------------------

    def set_ghost(
        self,
        xml_path: str | Path,
        alpha: float = 0.3,
    ) -> None:
        """Load a semi-transparent "ghost" robot for A/B overlay.

        .. note::
            This is a placeholder stub.  Full implementation requires
            loading a second ``MjModel`` and adjusting geom ``rgba``
            alpha values.  Deferred to V1.

        Parameters
        ----------
        xml_path:
            MJCF path for the ghost robot (typically the same model).
        alpha:
            Transparency value in ``[0, 1]``.
        """
        # TODO(V1): implement ghost overlay via a second MjModel instance
        #           with all geom rgba[:, 3] set to `alpha`.
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_callback(self, key: int, engine: PlaybackEngine) -> None:
        """Handle keyboard input from the MuJoCo viewer.

        Key codes follow the GLFW convention used by MuJoCo's passive
        viewer.  See https://www.glfw.org/docs/latest/group__keys.html

        Parameters
        ----------
        key:
            GLFW key code as an integer.
        engine:
            Playback engine to control.
        """
        GLFW_KEY_SPACE = 32       # noqa: N806
        GLFW_KEY_LEFT = 263       # noqa: N806
        GLFW_KEY_RIGHT = 262      # noqa: N806
        GLFW_KEY_R = 82           # noqa: N806
        GLFW_KEY_1 = 49           # noqa: N806 (clips 1-9 = keys 49-57)
        GLFW_KEY_9 = 57           # noqa: N806
        GLFW_KEY_LBRACKET = 91    # noqa: N806  [  → speed ÷ 2
        GLFW_KEY_RBRACKET = 93    # noqa: N806  ]  → speed × 2

        if key == GLFW_KEY_SPACE:
            engine.toggle_play_pause()
        elif key == GLFW_KEY_LEFT:
            engine.step(-1)
        elif key == GLFW_KEY_RIGHT:
            engine.step(1)
        elif key == GLFW_KEY_R:
            engine.reset()
        elif GLFW_KEY_1 <= key <= GLFW_KEY_9:
            clip_idx = key - GLFW_KEY_1
            if clip_idx < engine.num_clips:
                engine.switch_clip(clip_idx)
        elif key == GLFW_KEY_LBRACKET:
            engine.set_speed(max(0.125, engine.speed / 2.0))
        elif key == GLFW_KEY_RBRACKET:
            engine.set_speed(min(8.0, engine.speed * 2.0))

    def _setup_camera(self, viewer: object) -> None:
        """Configure the default camera based on ``camera_mode``."""
        # MuJoCo viewer camera configuration API may vary across versions.
        # We attempt to set a sensible default and silently ignore failures.
        if self._camera_mode == "follow_root" and self._model is not None:
            try:
                # Try to find the 'pelvis' or first body after worldbody.
                mj = self._mj
                body_id = mj.mj_name2id(
                    self._model, mj.mjtObj.mjOBJ_BODY, "pelvis"
                )
                if body_id < 0:
                    body_id = 1  # first non-world body
                viewer.cam.lookat[:] = self._data.xpos[body_id]
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -20.0
                viewer.cam.azimuth = 90.0
            except Exception:  # noqa: BLE001
                pass

    def _run_headless(
        self,
        engine: PlaybackEngine,
        metrics_engine: object | None,
    ) -> None:
        """Run the playback loop without a display window (for testing)."""
        import numpy as np  # noqa: PLC0415

        mj = self._mj
        with mj.Renderer(self._model) as renderer:
            engine.play()
            n = engine.current_motion.motion_length
            for _ in range(n):
                engine.tick()
                renderer.update_scene(self._data)
                # Offscreen render (result discarded unless caller inspects).
                _ = renderer.render()
