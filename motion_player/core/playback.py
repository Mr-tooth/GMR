"""Playback engine for motion_player.

:class:`PlaybackEngine` implements a simple state machine that manages the
current frame index and dispatches ``on_frame_change`` callbacks to
registered backends.  It is intentionally free of any rendering or GUI
dependencies so it can be driven from a CLI, a test harness, or a GUI
event loop without modification.
"""

from __future__ import annotations

import time
from typing import Callable

from motion_player.core.models import StandardMotion


# Frame-change callback signature: (frame_idx, motion) -> None
FrameCallback = Callable[[int, StandardMotion], None]


class PlaybackEngine:
    """Frame-level playback state machine.

    The engine tracks the current frame index and knows whether playback is
    active.  It does **not** create any threads; callers must drive it by
    calling :meth:`tick` in a loop (typically the renderer's main loop).

    Parameters
    ----------
    motion:
        The :class:`~motion_player.core.models.StandardMotion` clip to play.
    fps_override:
        If provided, overrides ``motion.fps`` for playback pacing.  Useful
        for playing back at a different speed than the capture rate.
    loop:
        If ``True`` (default), playback wraps around to frame 0 after the
        last frame.

    Examples
    --------
    >>> engine = PlaybackEngine(motion)
    >>> engine.on_frame_change(lambda idx, m: print(f"frame {idx}"))
    >>> engine.play()
    >>> while True:
    ...     engine.tick()
    """

    # Maximum number of clips that can be loaded simultaneously.
    # Keeping this small prevents accidental memory exhaustion when a
    # user inadvertently passes a directory of hundreds of clips instead
    # of individual files.  Raise if your use-case requires more.
    _MAX_CLIPS = 10

    def __init__(
        self,
        motion: StandardMotion,
        fps_override: float | None = None,
        loop: bool = True,
    ) -> None:
        self._clips: list[StandardMotion] = [motion]
        self._active_clip_idx: int = 0
        self._frame_idx: int = 0
        self._is_playing: bool = False
        self._loop: bool = loop
        self._speed: float = 1.0
        self._callbacks: list[FrameCallback] = []
        self._fps_override = fps_override

        # Monotonic clock for real-time pacing.
        self._last_tick_time: float = time.monotonic()
        self._accumulated_dt: float = 0.0

    # ------------------------------------------------------------------
    # Clip management
    # ------------------------------------------------------------------

    def add_clip(self, motion: StandardMotion) -> int:
        """Add a clip and return its index.

        Parameters
        ----------
        motion:
            New clip to register.

        Returns
        -------
        int
            Index of the newly added clip.
        """
        if len(self._clips) >= self._MAX_CLIPS:
            raise RuntimeError(
                f"Cannot add more than {self._MAX_CLIPS} clips."
            )
        self._clips.append(motion)
        return len(self._clips) - 1

    def switch_clip(self, clip_idx: int) -> None:
        """Switch the active clip and reset to frame 0.

        Parameters
        ----------
        clip_idx:
            Zero-based clip index.
        """
        if not 0 <= clip_idx < len(self._clips):
            raise IndexError(
                f"Clip index {clip_idx} out of range [0, {len(self._clips)})."
            )
        self._active_clip_idx = clip_idx
        self._frame_idx = 0
        self._accumulated_dt = 0.0
        self._dispatch_callback()

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Start or resume playback."""
        self._is_playing = True
        self._last_tick_time = time.monotonic()
        self._accumulated_dt = 0.0

    def pause(self) -> None:
        """Pause playback (current frame stays)."""
        self._is_playing = False

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause."""
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def reset(self) -> None:
        """Reset to frame 0 and pause."""
        self._frame_idx = 0
        self._is_playing = False
        self._accumulated_dt = 0.0
        self._dispatch_callback()

    def step(self, delta: int = 1) -> None:
        """Advance (or retreat) by *delta* frames without playing.

        Parameters
        ----------
        delta:
            Number of frames to step.  Positive steps forward, negative
            steps backward.  Wraps around in loop mode.
        """
        n = self.current_motion.motion_length
        self._frame_idx = (self._frame_idx + delta) % n if self._loop else max(
            0, min(n - 1, self._frame_idx + delta)
        )
        self._dispatch_callback()

    def seek(self, frame_idx: int) -> None:
        """Jump to a specific frame index.

        Parameters
        ----------
        frame_idx:
            Target frame (clamped to ``[0, motion_length - 1]``).
        """
        n = self.current_motion.motion_length
        self._frame_idx = max(0, min(n - 1, frame_idx))
        self._dispatch_callback()

    # ------------------------------------------------------------------
    # Speed control
    # ------------------------------------------------------------------

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier.

        Parameters
        ----------
        speed:
            Multiplier relative to real-time.  ``1.0`` = normal speed,
            ``0.5`` = half speed, ``2.0`` = double speed.

        Raises
        ------
        ValueError
            If *speed* is not positive.
        """
        if speed <= 0:
            raise ValueError(f"Playback speed must be > 0, got {speed}.")
        self._speed = speed

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_frame_change(self, callback: FrameCallback) -> None:
        """Register a callback to be called whenever the current frame changes.

        The callback receives ``(frame_idx: int, motion: StandardMotion)``
        as arguments.  Multiple callbacks can be registered; they are called
        in registration order.

        Parameters
        ----------
        callback:
            Callable with signature ``(int, StandardMotion) -> None``.
        """
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # Main loop integration
    # ------------------------------------------------------------------

    def tick(self) -> bool:
        """Advance the playback clock by one wall-clock step.

        Call this method once per iteration of your main render loop.
        When playing, it accumulates real time and advances the frame
        index whenever enough time has elapsed for the next frame.

        Returns
        -------
        bool
            ``True`` if the frame index changed (useful for triggering
            renders only when necessary).
        """
        if not self._is_playing:
            return False

        now = time.monotonic()
        elapsed = now - self._last_tick_time
        self._last_tick_time = now
        self._accumulated_dt += elapsed * self._speed

        fps = self._fps_override or self.current_motion.fps
        frame_duration = 1.0 / fps

        changed = False
        while self._accumulated_dt >= frame_duration:
            self._accumulated_dt -= frame_duration
            n = self.current_motion.motion_length

            if self._frame_idx >= n - 1:
                if self._loop:
                    self._frame_idx = 0
                else:
                    self._is_playing = False
                    self._accumulated_dt = 0.0
                    break
            else:
                self._frame_idx += 1

            self._dispatch_callback()
            changed = True

        return changed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_frame(self) -> int:
        """Current frame index (0-based)."""
        return self._frame_idx

    @property
    def is_playing(self) -> bool:
        """``True`` if playback is currently active."""
        return self._is_playing

    @property
    def speed(self) -> float:
        """Current playback speed multiplier."""
        return self._speed

    @property
    def loop(self) -> bool:
        """``True`` if playback loops at the end of the clip."""
        return self._loop

    @loop.setter
    def loop(self, value: bool) -> None:
        self._loop = value

    @property
    def current_motion(self) -> StandardMotion:
        """The currently active :class:`~motion_player.core.models.StandardMotion`."""
        return self._clips[self._active_clip_idx]

    @property
    def num_clips(self) -> int:
        """Number of loaded clips."""
        return len(self._clips)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_callback(self) -> None:
        """Invoke all registered frame-change callbacks."""
        motion = self.current_motion
        for cb in self._callbacks:
            cb(self._frame_idx, motion)
