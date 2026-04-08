"""Data models for motion_player.

All dataclasses defined here are the *canonical* in-memory representation
used throughout the motion_player package.  Heavy numerical libraries
(numpy) are imported unconditionally because they are a hard dependency;
everything else is imported lazily or under ``TYPE_CHECKING``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # forward-reference type hints only


# ---------------------------------------------------------------------------
# StandardMotion
# ---------------------------------------------------------------------------

@dataclass
class StandardMotion:
    """In-memory representation of a standardised robot motion clip.

    All arrays are ``numpy.float32``.  Quaternions use **xyzw (scalar-last)**
    convention throughout this package; backends are responsible for
    converting to their own convention (e.g. MuJoCo uses wxyz internally).

    The ``motion_length`` field equals ``N - 1`` (the AMP N-1 temporal
    alignment semantics from ``rsl-rl-ex``): every array has ``N_eff`` rows
    where ``N_eff == motion_length``.

    Parameters
    ----------
    fps : float
        Capture / playback frame rate.
    motion_length : int
        Number of usable frames (= original N − 1).
    motion_weight : float
        Sampling weight for this clip (used by the AMP motion loader).
    root_pos : np.ndarray, shape ``(N_eff, 3)``
        Root (pelvis) position in world frame, t₀ frames.
    root_rot : np.ndarray, shape ``(N_eff, 4)``
        Root rotation quaternion **xyzw**, t₀ frames.
    projected_gravity : np.ndarray, shape ``(N_eff, 3)``
        World gravity ``[0, 0, -1]`` projected to root-local frame.
    root_lin_vel : np.ndarray, shape ``(N_eff, 3)``
        Root linear velocity in root-local frame (finite difference).
    root_ang_vel : np.ndarray, shape ``(N_eff, 3)``
        Root angular velocity in root-local frame (rotvec / dt).
    dof_pos : np.ndarray, shape ``(N_eff, num_dofs)``
        Joint positions (radians), t₀ frames.
    dof_vel : np.ndarray, shape ``(N_eff, num_dofs)``
        Joint velocities (rad/s), finite difference.
    key_body_pos_local : np.ndarray, shape ``(N_eff, K * 3)``
        Positions of *all* bodies in root-local frame, flattened.
        Downstream code can subset to end-effectors as needed.
    dof_names : list of str or None
        Ordered DOF names (length ``num_dofs``).  Populated from a sidecar
        YAML if available; ``None`` otherwise.
    key_body_names : list of str or None
        Ordered body names (length ``K``).  Populated from a sidecar YAML.
    robot : str or None
        Robot identifier string, e.g. ``"booster_t1"``.
    source_pipeline : str or None
        Free-form description of the retargeting pipeline used to produce
        this clip, e.g. ``"gmr_bvh_lafan1"``.
    gmr_ik_config : str or None
        Filename of the GMR IK config JSON used, e.g.
        ``"bvh_lafan1_to_t1_29dof"``.
    """

    fps: float
    motion_length: int
    motion_weight: float

    root_pos: np.ndarray            # (N_eff, 3)
    root_rot: np.ndarray            # (N_eff, 4)  xyzw
    projected_gravity: np.ndarray   # (N_eff, 3)
    root_lin_vel: np.ndarray        # (N_eff, 3)
    root_ang_vel: np.ndarray        # (N_eff, 3)
    dof_pos: np.ndarray             # (N_eff, num_dofs)
    dof_vel: np.ndarray             # (N_eff, num_dofs)
    key_body_pos_local: np.ndarray  # (N_eff, K*3)

    # Optional metadata populated from sidecar YAML
    dof_names: list[str] | None = field(default=None)
    key_body_names: list[str] | None = field(default=None)
    robot: str | None = field(default=None)
    source_pipeline: str | None = field(default=None)
    gmr_ik_config: str | None = field(default=None)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_dofs(self) -> int:
        """Number of controllable degrees of freedom."""
        return int(self.dof_pos.shape[1])

    @property
    def num_key_bodies(self) -> int:
        """Number of key bodies tracked in ``key_body_pos_local``."""
        return int(self.key_body_pos_local.shape[1]) // 3

    @property
    def dt(self) -> float:
        """Time step between consecutive frames (seconds)."""
        return 1.0 / self.fps

    def __post_init__(self) -> None:
        """Validate array shapes after construction."""
        n = self.motion_length
        self._validate_shape("root_pos", self.root_pos, (n, 3))
        self._validate_shape("root_rot", self.root_rot, (n, 4))
        self._validate_shape("projected_gravity", self.projected_gravity, (n, 3))
        self._validate_shape("root_lin_vel", self.root_lin_vel, (n, 3))
        self._validate_shape("root_ang_vel", self.root_ang_vel, (n, 3))
        # dof_pos / dof_vel share the same shape but num_dofs can vary.
        if self.dof_pos.shape[0] != n:
            raise ValueError(
                f"dof_pos.shape[0]={self.dof_pos.shape[0]} != "
                f"motion_length={n}"
            )
        if self.dof_vel.shape != self.dof_pos.shape:
            raise ValueError(
                f"dof_vel.shape={self.dof_vel.shape} != "
                f"dof_pos.shape={self.dof_pos.shape}"
            )
        if self.key_body_pos_local.shape[0] != n:
            raise ValueError(
                f"key_body_pos_local.shape[0]={self.key_body_pos_local.shape[0]}"
                f" != motion_length={n}"
            )

    @staticmethod
    def _validate_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
        if arr.shape != expected:
            raise ValueError(
                f"{name}.shape={arr.shape} != expected {expected}"
            )


# ---------------------------------------------------------------------------
# RobotModel
# ---------------------------------------------------------------------------

@dataclass
class RobotModel:
    """Lightweight description of a robot model loaded for playback.

    This dataclass stores the information needed to map a :class:`StandardMotion`
    frame to a MuJoCo ``qpos`` vector.  It is backend-agnostic; the actual
    MuJoCo ``MjModel`` object is stored separately inside the backend.

    Parameters
    ----------
    name : str
        Robot identifier (e.g. ``"booster_t1"``).
    mjcf_path : str
        Absolute path to the MJCF XML file.
    root_joint_name : str
        Name of the free joint that represents the floating base.
    dof_qpos_indices : list of int
        Mapping from dataset DOF column index → MuJoCo ``qpos`` index
        (after the 7 free-joint entries).
    sign_flip : np.ndarray, shape ``(num_dofs,)``
        Per-DOF sign correction factor (+1 or −1).
    offset : np.ndarray, shape ``(num_dofs,)``
        Per-DOF zero-position offset in radians.
    jnt_range : np.ndarray or None, shape ``(num_dofs, 2)``
        Per-DOF ``[lower, upper]`` joint limits in radians (from MJCF).
        ``None`` if limits are not defined or model is not yet loaded.
    """

    name: str
    mjcf_path: str
    root_joint_name: str
    dof_qpos_indices: list[int]
    sign_flip: np.ndarray                    # (num_dofs,)
    offset: np.ndarray                       # (num_dofs,)
    jnt_range: np.ndarray | None = field(default=None)  # (num_dofs, 2)

    @property
    def num_dofs(self) -> int:
        """Number of actuated DOFs (excluding the 6-DOF floating base)."""
        return len(self.dof_qpos_indices)


# ---------------------------------------------------------------------------
# EditState  (undo / redo stack entry)
# ---------------------------------------------------------------------------

@dataclass
class EditState:
    """A single edit operation snapshot used for undo / redo.

    Parameters
    ----------
    frame_idx : int
        Index of the edited frame (0-based).
    field : str
        Name of the ``StandardMotion`` field that was modified.
        One of ``"root_pos"``, ``"root_rot"``, ``"dof_pos"``.
    before : np.ndarray
        Array value *before* the edit (copy).
    after : np.ndarray
        Array value *after* the edit (copy).
    """

    frame_idx: int
    field: str
    before: np.ndarray
    after: np.ndarray


# ---------------------------------------------------------------------------
# DOF Audit Report
# ---------------------------------------------------------------------------

@dataclass
class DOFAuditReport:
    """Result of comparing dataset DOF order against a robot model.

    Parameters
    ----------
    matched : list of str
        DOF names that match in both name and order.
    mismatched : list of tuple[str, str]
        ``(dataset_name, model_name)`` pairs where names differ at the
        same column index.
    unmatched_in_data : list of str
        DOF names present in the dataset but absent in the model.
    unmatched_in_model : list of str
        DOF names present in the model but absent in the dataset.
    is_order_compatible : bool
        ``True`` if all names match and ordering is consistent.
    """

    matched: list[str] = field(default_factory=list)
    mismatched: list[tuple[str, str]] = field(default_factory=list)
    unmatched_in_data: list[str] = field(default_factory=list)
    unmatched_in_model: list[str] = field(default_factory=list)
    is_order_compatible: bool = False
