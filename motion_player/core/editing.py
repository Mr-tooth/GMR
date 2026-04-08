"""Editing engine for motion_player.

:class:`EditingEngine` provides frame-level and segment-level editing
operations on a :class:`~motion_player.core.models.StandardMotion` clip,
together with an undo / redo stack.

All operations mutate the motion **in-place** (copying the old value into
the undo stack beforehand).  The engine does *not* depend on any rendering
backend — it only knows about ``StandardMotion`` data.

Supported operations
--------------------
Frame edits (immediate, single frame):
  - :meth:`edit_root_pos`    — root XYZ translation delta
  - :meth:`edit_root_rot_euler` — root rotation delta (Roll/Pitch/Yaw)
  - :meth:`edit_dof`         — single DOF angle delta

Segment edits (over a frame range ``[i0, i1]``):
  - :meth:`interpolate_segment` — linear or Catmull-Rom interpolation
  - :meth:`smooth_segment`      — Savitzky–Golay or low-pass filter

Cross-frame propagation:
  - :meth:`propagate` — apply a weighted delta to subsequent N frames

Undo / redo:
  - :meth:`undo`, :meth:`redo`
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore[import]

from motion_player.core.models import EditState, RobotModel, StandardMotion


class EditingEngine:
    """Apply frame-level and segment-level edits to a ``StandardMotion``.

    Parameters
    ----------
    motion:
        The motion clip to edit.  Edits are applied in-place.
    robot_model:
        Optional robot model used for joint-limit clamping after edits.
        If ``None``, joint limits are not enforced.
    undo_stack_size:
        Maximum number of undoable operations (default 50).
    recompute_vel:
        If ``True``, ``dof_vel`` and root velocity fields are recomputed
        via finite differences after every edit.  This keeps derived
        fields consistent but adds computation per edit.  Default ``False``
        (velocities are left as-is and can be recomputed on export).
    """

    def __init__(
        self,
        motion: StandardMotion,
        robot_model: RobotModel | None = None,
        undo_stack_size: int = 50,
        recompute_vel: bool = False,
    ) -> None:
        self._motion = motion
        self._robot_model = robot_model
        self._undo_stack_size = undo_stack_size
        self._recompute_vel = recompute_vel
        self._undo_stack: list[EditState] = []
        self._redo_stack: list[EditState] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def motion(self) -> StandardMotion:
        """The motion clip being edited (possibly modified in-place)."""
        return self._motion

    # ------------------------------------------------------------------
    # Frame-level edits
    # ------------------------------------------------------------------

    def edit_root_pos(self, frame_idx: int, delta: np.ndarray) -> None:
        """Add *delta* (shape ``(3,)``) to ``root_pos[frame_idx]``.

        Parameters
        ----------
        frame_idx:
            Target frame index (0-based).
        delta:
            XYZ displacement in metres.
        """
        self._check_frame(frame_idx)
        before = self._motion.root_pos[frame_idx].copy()
        self._motion.root_pos[frame_idx] += np.asarray(delta, dtype=np.float32)
        after = self._motion.root_pos[frame_idx].copy()
        self._push_undo(EditState(frame_idx, "root_pos", before, after))

    def edit_root_rot_euler(
        self, frame_idx: int, delta_rpy: np.ndarray
    ) -> None:
        """Compose a rotation delta (Roll, Pitch, Yaw in radians) into
        ``root_rot[frame_idx]``.

        The delta is applied as a *right-multiplication* of the current
        quaternion:  ``q_new = q_cur * q_delta``, where ``q_delta``
        corresponds to the given Euler angles (intrinsic XYZ / RPY).
        The result is normalised.

        Parameters
        ----------
        frame_idx:
            Target frame index.
        delta_rpy:
            Roll / Pitch / Yaw increments in radians, shape ``(3,)``.
        """
        self._check_frame(frame_idx)
        before = self._motion.root_rot[frame_idx].copy()

        q_cur = R.from_quat(self._motion.root_rot[frame_idx])  # xyzw
        q_delta = R.from_euler("xyz", delta_rpy)
        q_new = (q_cur * q_delta).as_quat()  # xyzw, normalised by scipy

        self._motion.root_rot[frame_idx] = q_new.astype(np.float32)
        self._push_undo(
            EditState(frame_idx, "root_rot", before, q_new.copy())
        )

    def edit_dof(
        self, frame_idx: int, dof_idx: int, delta: float
    ) -> None:
        """Add *delta* (radians) to ``dof_pos[frame_idx, dof_idx]``.

        After the edit the value is clamped to the joint's limit range
        if a :class:`~motion_player.core.models.RobotModel` with
        ``jnt_range`` was provided.

        Parameters
        ----------
        frame_idx:
            Target frame index.
        dof_idx:
            Column index in ``dof_pos``.
        delta:
            Angle increment in radians.
        """
        self._check_frame(frame_idx)
        n_dofs = self._motion.num_dofs
        if not 0 <= dof_idx < n_dofs:
            raise IndexError(f"dof_idx {dof_idx} out of range [0, {n_dofs}).")

        before = float(self._motion.dof_pos[frame_idx, dof_idx])
        self._motion.dof_pos[frame_idx, dof_idx] += delta
        self._clamp_dof(frame_idx, dof_idx)
        after = float(self._motion.dof_pos[frame_idx, dof_idx])
        self._push_undo(
            EditState(
                frame_idx,
                "dof_pos",
                np.array([before], dtype=np.float32),
                np.array([after], dtype=np.float32),
            )
        )

    # ------------------------------------------------------------------
    # Segment edits
    # ------------------------------------------------------------------

    def interpolate_segment(
        self,
        i0: int,
        i1: int,
        method: Literal["linear", "catmull_rom"] = "linear",
        fields: list[str] | None = None,
    ) -> None:
        """Interpolate frames between *i0* and *i1* (endpoints kept fixed).

        Parameters
        ----------
        i0, i1:
            Start and end frame indices (inclusive).  The values at
            ``i0`` and ``i1`` are used as boundary conditions.
        method:
            ``"linear"`` (default) or ``"catmull_rom"`` (smooth spline).
        fields:
            List of ``StandardMotion`` fields to interpolate.  Defaults
            to ``["root_pos", "root_rot", "dof_pos"]``.
        """
        if fields is None:
            fields = ["root_pos", "root_rot", "dof_pos"]
        self._check_range(i0, i1)
        n_inner = i1 - i0 - 1
        if n_inner <= 0:
            return  # nothing to interpolate

        for field in fields:
            arr = getattr(self._motion, field)
            v0, v1 = arr[i0].copy(), arr[i1].copy()

            if field == "root_rot":
                # Use SLERP for quaternions.
                for k in range(1, i1 - i0):
                    t = k / (i1 - i0)
                    arr[i0 + k] = self._slerp(v0, v1, t)
            else:
                for k in range(1, i1 - i0):
                    t = k / (i1 - i0)
                    arr[i0 + k] = (1.0 - t) * v0 + t * v1

    def smooth_segment(
        self,
        i0: int,
        i1: int,
        method: Literal["savgol", "lowpass"] = "savgol",
        window: int = 11,
        polyorder: int = 3,
        fields: list[str] | None = None,
    ) -> None:
        """Apply a smoothing filter to frames in ``[i0, i1]``.

        Parameters
        ----------
        i0, i1:
            Start and end frame indices (inclusive).
        method:
            ``"savgol"`` (Savitzky–Golay, default) or ``"lowpass"``
            (Butterworth low-pass via ``scipy.signal``).
        window:
            Window length for Savitzky–Golay (must be odd and ≥ 1).
            For low-pass, this is the filter order.
        polyorder:
            Polynomial order for Savitzky–Golay (must be < ``window``).
        fields:
            Fields to smooth.  Defaults to ``["dof_pos", "root_pos"]``.
        """
        from scipy.signal import savgol_filter, butter, filtfilt  # lazy

        if fields is None:
            fields = ["dof_pos", "root_pos"]
        self._check_range(i0, i1)
        seg_len = i1 - i0 + 1

        for field in fields:
            arr = getattr(self._motion, field)
            seg = arr[i0 : i1 + 1].copy()  # (seg_len, D)

            if method == "savgol":
                win = min(window, seg_len)
                if win % 2 == 0:
                    win -= 1
                win = max(win, 3)
                po = min(polyorder, win - 1)
                smoothed = savgol_filter(seg, win, po, axis=0)
            elif method == "lowpass":
                if seg_len < 10:
                    continue  # too short for filter
                b, a = butter(window, 0.3, btype="low")
                smoothed = filtfilt(b, a, seg, axis=0)
            else:
                raise ValueError(f"Unknown smoothing method {method!r}.")

            arr[i0 : i1 + 1] = smoothed.astype(arr.dtype)

        # Re-normalise quaternions if root_rot was smoothed.
        if "root_rot" in fields:
            norms = np.linalg.norm(
                self._motion.root_rot[i0 : i1 + 1], axis=-1, keepdims=True
            )
            norms = np.where(norms < 1e-8, 1.0, norms)
            self._motion.root_rot[i0 : i1 + 1] /= norms

    # ------------------------------------------------------------------
    # Cross-frame propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        frame_idx: int,
        field: str,
        delta: np.ndarray,
        n_frames: int,
        decay: Literal["linear", "cosine", "constant"] = "linear",
    ) -> None:
        """Apply a weighted *delta* to *n_frames* frames starting at *frame_idx*.

        The delta is applied as::

            motion[frame_idx + k].field += weight(k) * delta

        where ``weight`` decreases from 1.0 (at ``k=0``) to 0.0 (at
        ``k=n_frames``) according to *decay*.

        Parameters
        ----------
        frame_idx:
            Starting frame index.
        field:
            ``StandardMotion`` field name to modify.
            One of ``"root_pos"``, ``"root_rot"``, ``"dof_pos"``.
        delta:
            Additive delta to propagate.  Shape must broadcast with the
            field's per-frame slice.
        n_frames:
            Number of frames to propagate into (including *frame_idx*).
        decay:
            ``"linear"`` (default), ``"cosine"``, or ``"constant"``.
        """
        if not hasattr(self._motion, field):
            raise AttributeError(f"StandardMotion has no field {field!r}.")

        n_total = self._motion.motion_length
        end = min(frame_idx + n_frames, n_total)
        arr = getattr(self._motion, field)
        delta = np.asarray(delta, dtype=np.float32)

        for k, i in enumerate(range(frame_idx, end)):
            t = k / max(n_frames - 1, 1)
            if decay == "linear":
                weight = 1.0 - t
            elif decay == "cosine":
                weight = 0.5 * (1.0 + np.cos(np.pi * t))
            elif decay == "constant":
                weight = 1.0
            else:
                raise ValueError(f"Unknown decay {decay!r}.")

            if field == "root_rot":
                # Compose as a scaled rotation, not additive.
                q_cur = R.from_quat(arr[i])
                axis_angle = R.from_quat(delta).as_rotvec() * weight
                q_delta = R.from_rotvec(axis_angle)
                arr[i] = (q_cur * q_delta).as_quat().astype(np.float32)
            else:
                arr[i] += weight * delta

        # Post-process: normalise quats and clamp DOF limits.
        if field == "root_rot":
            norms = np.linalg.norm(arr[frame_idx:end], axis=-1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            arr[frame_idx:end] /= norms
        elif field == "dof_pos" and self._robot_model is not None:
            self._clamp_dof_range(frame_idx, end)

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def undo(self) -> bool:
        """Undo the most recent edit.

        Returns
        -------
        bool
            ``True`` if an operation was undone; ``False`` if the stack is
            empty.
        """
        if not self._undo_stack:
            return False
        state = self._undo_stack.pop()
        self._apply_edit_state(state, use_before=True)
        self._redo_stack.append(state)
        return True

    def redo(self) -> bool:
        """Redo the most recently undone edit.

        Returns
        -------
        bool
            ``True`` if an operation was redone; ``False`` if the redo
            stack is empty.
        """
        if not self._redo_stack:
            return False
        state = self._redo_stack.pop()
        self._apply_edit_state(state, use_before=False)
        self._undo_stack.append(state)
        return True

    # ------------------------------------------------------------------
    # Velocity recomputation (optional post-processing)
    # ------------------------------------------------------------------

    def recompute_velocities(self) -> None:
        """Recompute ``dof_vel`` and root velocity fields from positions.

        This is a potentially expensive operation (O(N × D)) and is
        intended to be called on export rather than after every edit.
        """
        fps = self._motion.fps
        self._motion.dof_vel[:] = np.diff(
            self._motion.dof_pos, axis=0, append=self._motion.dof_pos[-1:]
        ) * fps

        # Root linear velocity (world frame, finite difference of root_pos).
        lin_vel = np.diff(
            self._motion.root_pos, axis=0, append=self._motion.root_pos[-1:]
        ) * fps
        self._motion.root_lin_vel[:] = lin_vel

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_undo(self, state: EditState) -> None:
        """Push *state* onto the undo stack (clearing the redo stack)."""
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._undo_stack_size:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _apply_edit_state(
        self, state: EditState, use_before: bool
    ) -> None:
        """Apply a saved edit state (for undo/redo)."""
        arr = getattr(self._motion, state.field)
        value = state.before if use_before else state.after
        if state.field == "dof_pos":
            # For scalar dof edits we stored a single-element array.
            # Restore only if shapes differ (simple heuristic).
            if value.shape == arr[state.frame_idx].shape:
                arr[state.frame_idx] = value
        else:
            arr[state.frame_idx] = value

    def _check_frame(self, frame_idx: int) -> None:
        n = self._motion.motion_length
        if not 0 <= frame_idx < n:
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {n})."
            )

    def _check_range(self, i0: int, i1: int) -> None:
        self._check_frame(i0)
        self._check_frame(i1)
        if i0 >= i1:
            raise ValueError(f"i0={i0} must be < i1={i1}.")

    def _clamp_dof(self, frame_idx: int, dof_idx: int) -> None:
        """Clamp a single DOF to its joint limit."""
        if (
            self._robot_model is None
            or self._robot_model.jnt_range is None
        ):
            return
        lo, hi = self._robot_model.jnt_range[dof_idx]
        val = self._motion.dof_pos[frame_idx, dof_idx]
        self._motion.dof_pos[frame_idx, dof_idx] = float(
            np.clip(val, lo, hi)
        )

    def _clamp_dof_range(self, i0: int, i1: int) -> None:
        """Clamp all DOFs in frame range ``[i0, i1)`` to joint limits."""
        if (
            self._robot_model is None
            or self._robot_model.jnt_range is None
        ):
            return
        lo = self._robot_model.jnt_range[:, 0]
        hi = self._robot_model.jnt_range[:, 1]
        self._motion.dof_pos[i0:i1] = np.clip(
            self._motion.dof_pos[i0:i1], lo, hi
        )

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two unit quaternions.

        Parameters
        ----------
        q0, q1:
            Unit quaternions in xyzw order.
        t:
            Interpolation parameter in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Interpolated unit quaternion (xyzw).
        """
        r = R.slerp(  # type: ignore[attr-defined]
            [0.0, 1.0], R.from_quat([q0, q1])
        )
        return r(t).as_quat().astype(np.float32)
