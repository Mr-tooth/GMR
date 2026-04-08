"""Dataset and model adapters for motion_player.

This module provides:

- :class:`DatasetAdapter` — load / save standard motion files (``.pkl`` /
  ``.npy``) and populate :class:`~motion_player.core.models.StandardMotion`.
- :class:`ModelAdapter` — load a robot MJCF and build the DOF-to-qpos
  mapping needed to drive MuJoCo from a ``StandardMotion`` frame.
- :class:`DOFAuditor` — compare dataset DOF order against a robot model
  and optionally reorder columns to match the canonical order.

Heavy dependencies (``mujoco``, ``yaml``) are imported lazily inside
methods so that the module is safe to import in environments where those
libraries are not installed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from motion_player.core.models import (
    DOFAuditReport,
    RobotModel,
    StandardMotion,
)

# ---------------------------------------------------------------------------
# Required keys in a standard motion file.
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = frozenset({
    "fps",
    "root_pos",
    "root_rot",
    "dof_pos",
})

# All keys that constitute a full standard motion record.
_STANDARD_KEYS = frozenset({
    "fps",
    "motion_length",
    "motion_weight",
    "root_pos",
    "root_rot",
    "projected_gravity",
    "root_lin_vel",
    "root_ang_vel",
    "dof_pos",
    "dof_vel",
    "key_body_pos_local",
})


# ---------------------------------------------------------------------------
# DatasetAdapter
# ---------------------------------------------------------------------------

class DatasetAdapter:
    """Load and save standard motion files.

    Standard files are produced by ``rsl-rl-ex``'s ``data_builder.py`` and
    stored as Python pickles or ``.npy`` dicts.  The schema is documented in
    ``docs/requirements.md §6.1``.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        path: str | Path,
        sidecar_path: str | Path | None = None,
    ) -> StandardMotion:
        """Load a single clip from *path*.

        Parameters
        ----------
        path:
            Path to a ``*_standard.pkl`` or ``.npy`` file.
        sidecar_path:
            Optional path to a ``*_meta.yaml`` sidecar that provides
            ``dof_names``, ``key_body_names``, and other metadata.
            If ``None``, the loader will look for
            ``<stem>_meta.yaml`` next to *path*.

        Returns
        -------
        StandardMotion
            Validated and normalised in-memory representation.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        KeyError
            If a required field is missing from the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Motion file not found: {path}")

        raw = self._read_raw(path)
        self._validate_required_keys(raw, path)
        motion = self._build_standard_motion(raw)

        # Attempt to load sidecar metadata.
        sidecar = sidecar_path or path.with_name(path.stem + "_meta.yaml")
        meta = self._load_sidecar(sidecar)
        if meta:
            motion.dof_names = meta.get("dof_names")
            motion.key_body_names = meta.get("key_body_names")
            motion.robot = meta.get("robot")
            motion.source_pipeline = meta.get("source_pipeline")
            motion.gmr_ik_config = meta.get("gmr_ik_config")

        return motion

    def load_dataset(self, directory: str | Path) -> list[StandardMotion]:
        """Load all standard motion clips from *directory*.

        Scans for ``*_standard.pkl`` and ``*_standard.npy`` files.

        Parameters
        ----------
        directory:
            Path to a directory containing standard motion files.

        Returns
        -------
        list of StandardMotion
            One entry per successfully loaded clip (errors are logged and
            skipped rather than raising).
        """
        directory = Path(directory)
        clips: list[StandardMotion] = []
        patterns = ["*_standard.pkl", "*_standard.npy", "*.pkl", "*.npy"]
        seen: set[Path] = set()

        for pattern in patterns:
            for p in sorted(directory.glob(pattern)):
                if p in seen:
                    continue
                seen.add(p)
                try:
                    clips.append(self.load(p))
                except Exception as exc:  # noqa: BLE001
                    print(f"[DatasetAdapter] Skipping {p.name}: {exc}")

        return clips

    def save(self, motion: StandardMotion, path: str | Path) -> None:
        """Save *motion* to *path* in the standard ``.pkl`` format.

        The output is compatible with the ``motion_loader`` in
        ``rsl-rl-ex``: the dict layout mirrors the original
        ``standard_data`` structure produced by ``data_builder.py``.

        Parameters
        ----------
        motion:
            The :class:`StandardMotion` to serialise.
        path:
            Destination path.  The ``.pkl`` extension is added if absent.
        """
        path = Path(path)
        if path.suffix not in {".pkl", ".npy"}:
            path = path.with_suffix(".pkl")

        data: dict[str, Any] = {
            "fps": motion.fps,
            "motion_length": motion.motion_length,
            "motion_weight": motion.motion_weight,
            "root_pos": motion.root_pos,
            "root_rot": motion.root_rot,
            "projected_gravity": motion.projected_gravity,
            "root_lin_vel": motion.root_lin_vel,
            "root_ang_vel": motion.root_ang_vel,
            "dof_pos": motion.dof_pos,
            "dof_vel": motion.dof_vel,
            "key_body_pos_local": motion.key_body_pos_local,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_raw(path: Path) -> dict:
        """Read a raw dict from a ``.pkl`` or ``.npy`` file."""
        if path.suffix == ".npy":
            return dict(np.load(path, allow_pickle=True).item())
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301

    @staticmethod
    def _validate_required_keys(raw: dict, path: Path) -> None:
        missing = _REQUIRED_KEYS - set(raw.keys())
        if missing:
            raise KeyError(
                f"Standard motion file {path.name} is missing required keys: "
                f"{sorted(missing)}"
            )

    @staticmethod
    def _build_standard_motion(raw: dict) -> StandardMotion:
        """Convert a raw dict into a ``StandardMotion``, filling defaults."""
        fps = float(raw["fps"])
        root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
        root_rot = np.asarray(raw["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(raw["dof_pos"], dtype=np.float32)

        n_eff = root_pos.shape[0]

        motion_length = int(raw.get("motion_length", n_eff))
        motion_weight = float(raw.get("motion_weight", 1.0))

        projected_gravity = np.asarray(
            raw.get("projected_gravity", np.zeros((n_eff, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        root_lin_vel = np.asarray(
            raw.get("root_lin_vel", np.zeros((n_eff, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        root_ang_vel = np.asarray(
            raw.get("root_ang_vel", np.zeros((n_eff, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        dof_vel = np.asarray(
            raw.get("dof_vel", np.zeros_like(dof_pos)),
            dtype=np.float32,
        )
        key_body_pos_local = np.asarray(
            raw.get("key_body_pos_local", np.zeros((n_eff, 12), dtype=np.float32)),
            dtype=np.float32,
        )

        # Normalise root_rot quaternions.
        norms = np.linalg.norm(root_rot, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        root_rot = root_rot / norms

        return StandardMotion(
            fps=fps,
            motion_length=motion_length,
            motion_weight=motion_weight,
            root_pos=root_pos,
            root_rot=root_rot,
            projected_gravity=projected_gravity,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            key_body_pos_local=key_body_pos_local,
        )

    @staticmethod
    def _load_sidecar(sidecar_path: Path) -> dict | None:
        """Load a YAML sidecar file if it exists."""
        if not sidecar_path.exists():
            return None
        try:
            import yaml  # optional dependency
            with open(sidecar_path) as fh:
                return yaml.safe_load(fh) or {}
        except ImportError:
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"[DatasetAdapter] Failed to load sidecar {sidecar_path}: {exc}")
            return None


# ---------------------------------------------------------------------------
# ModelAdapter
# ---------------------------------------------------------------------------

class ModelAdapter:
    """Load a robot MJCF and build the DOF-to-qpos mapping.

    This adapter is responsible for translating a ``StandardMotion`` frame
    into the ``qpos`` array expected by MuJoCo.  The mapping is configured
    via a ``mapping.yaml`` file (schema documented in ``docs/design.md §4.1``).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_mapping(mapping_config_path: str | Path) -> dict:
        """Load and return the raw mapping config dict from a YAML file.

        Parameters
        ----------
        mapping_config_path:
            Path to ``mapping.yaml``.

        Returns
        -------
        dict
            Raw mapping configuration dictionary.
        """
        import yaml  # soft dependency

        with open(mapping_config_path) as fh:
            return yaml.safe_load(fh) or {}

    def load_mjcf(
        self,
        xml_path: str | Path,
        mapping_config: dict | None = None,
    ) -> RobotModel:
        """Load a MuJoCo MJCF and construct a :class:`RobotModel`.

        This method reads the MJCF, enumerates joints, and applies the
        configuration in *mapping_config* to build:

        - ``dof_qpos_indices``: dataset column index → MuJoCo qpos index
        - ``sign_flip``, ``offset``: per-DOF corrections
        - ``jnt_range``: joint limits from MJCF

        Parameters
        ----------
        xml_path:
            Path to the robot MJCF file.
        mapping_config:
            Parsed ``mapping.yaml`` dict.  If ``None``, the identity
            mapping is used (dataset columns assumed to match MJCF order).

        Returns
        -------
        RobotModel
            Populated robot model descriptor.
        """
        import mujoco as mj  # lazy import

        xml_path = Path(xml_path)
        model = mj.MjModel.from_xml_path(str(xml_path))

        cfg = mapping_config or {}
        root_joint_name: str = cfg.get("root_joint_name", "root")
        name_map: dict[str, str] = cfg.get("name_map", {})
        sign_flip_cfg: dict[str, float] = cfg.get("sign_flip", {})
        offset_cfg: dict[str, float] = cfg.get("offset", {})

        # Build {joint_name: qpos_adr} for all non-free joints.
        mjcf_joint_map: dict[str, int] = {}
        for jnt_id in range(model.njnt):
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id) or ""
            jtype = model.jnt_type[jnt_id]
            if jtype == mj.mjtJoint.mjJNT_FREE:
                continue  # skip floating base
            mjcf_joint_map[jname] = int(model.jnt_qposadr[jnt_id])

        # Determine dataset DOF order.
        dataset_dof_order: list[str] = cfg.get("dof_order_in_dataset", [])
        if not dataset_dof_order:
            # Fall back to MJCF order.
            dataset_dof_order = list(mjcf_joint_map.keys())

        num_dofs = len(dataset_dof_order)
        dof_qpos_indices: list[int] = []
        sign_flip = np.ones(num_dofs, dtype=np.float32)
        offset = np.zeros(num_dofs, dtype=np.float32)

        for col_idx, data_name in enumerate(dataset_dof_order):
            # Apply optional name remapping.
            model_name = name_map.get(data_name, data_name)
            qpos_adr = mjcf_joint_map.get(model_name)
            if qpos_adr is None:
                # Unmapped joint: map to index 7 + col_idx (best-effort).
                qpos_adr = 7 + col_idx
            dof_qpos_indices.append(qpos_adr)
            sign_flip[col_idx] = float(sign_flip_cfg.get(data_name, 1.0))
            offset[col_idx] = float(offset_cfg.get(data_name, 0.0))

        # Extract joint limits.
        jnt_range = self._extract_jnt_range(model, dataset_dof_order, name_map)

        return RobotModel(
            name=cfg.get("robot", xml_path.stem),
            mjcf_path=str(xml_path),
            root_joint_name=root_joint_name,
            dof_qpos_indices=dof_qpos_indices,
            sign_flip=sign_flip,
            offset=offset,
            jnt_range=jnt_range,
        )

    def motion_to_qpos(
        self,
        motion: StandardMotion,
        frame_idx: int,
        robot_model: RobotModel | None = None,
    ) -> np.ndarray:
        """Convert a ``StandardMotion`` frame to a MuJoCo ``qpos`` array.

        MuJoCo free-joint qpos layout::

            [x, y, z,  qw, qx, qy, qz,  dof_0, dof_1, ...]

        Note that MuJoCo uses **wxyz (scalar-first)** quaternion order,
        while ``StandardMotion.root_rot`` is **xyzw (scalar-last)**.
        This method handles the conversion automatically.

        Parameters
        ----------
        motion:
            Source motion clip.
        frame_idx:
            Frame index (0-based, within ``[0, motion_length)``.
        robot_model:
            If provided, DOF column remapping and sign/offset corrections
            are applied.  If ``None``, columns are written in dataset order.

        Returns
        -------
        np.ndarray
            ``qpos`` array suitable for assignment to ``mj_data.qpos``.
        """
        pos = motion.root_pos[frame_idx]   # (3,)
        rot_xyzw = motion.root_rot[frame_idx]  # (4,) xyzw

        # Convert xyzw → wxyz for MuJoCo.
        rot_wxyz = np.array(
            [rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]],
            dtype=np.float32,
        )

        dof = motion.dof_pos[frame_idx].copy()  # (num_dofs,)

        if robot_model is not None:
            # Apply per-DOF sign flip and zero-position offset.
            dof = dof * robot_model.sign_flip + robot_model.offset

        # Assemble qpos: [x, y, z, qw, qx, qy, qz, dof_0, ..., dof_N]
        qpos = np.concatenate([pos, rot_wxyz, dof]).astype(np.float32)
        return qpos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_jnt_range(
        model: Any,
        dataset_dof_order: list[str],
        name_map: dict[str, str],
    ) -> np.ndarray | None:
        """Extract per-DOF joint limits from the MuJoCo model.

        Parameters
        ----------
        model:
            ``mujoco.MjModel`` instance.
        dataset_dof_order:
            Ordered list of dataset DOF names.
        name_map:
            Optional name remapping dict (dataset → model).

        Returns
        -------
        np.ndarray or None
            Shape ``(num_dofs, 2)`` with ``[lo, hi]`` per DOF, or
            ``None`` if limits are unavailable.
        """
        try:
            import mujoco as mj

            n = len(dataset_dof_order)
            jnt_range = np.full((n, 2), [-np.inf, np.inf], dtype=np.float64)

            # Build {joint_name: jnt_id} for quick lookup.
            name_to_id = {}
            for jnt_id in range(model.njnt):
                jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id) or ""
                name_to_id[jname] = jnt_id

            for col_idx, data_name in enumerate(dataset_dof_order):
                model_name = name_map.get(data_name, data_name)
                jnt_id = name_to_id.get(model_name)
                if jnt_id is None:
                    continue
                if model.jnt_limited[jnt_id]:
                    jnt_range[col_idx] = model.jnt_range[jnt_id]

            return jnt_range.astype(np.float32)
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# DOFAuditor
# ---------------------------------------------------------------------------

class DOFAuditor:
    """Audit and repair DOF ordering between a dataset and a robot model.

    The standard motion files produced by ``rsl-rl-ex`` do not embed DOF
    names.  The ordering is implicitly inherited from the GMR retargeting
    pipeline (which follows MuJoCo XML import order).  This class helps
    detect and optionally repair ordering mismatches.
    """

    def audit(
        self,
        motion: StandardMotion,
        robot_model: RobotModel,
    ) -> DOFAuditReport:
        """Compare dataset DOF names against the robot model's joint order.

        Parameters
        ----------
        motion:
            Source motion clip with ``dof_names`` populated (from sidecar).
        robot_model:
            Robot model with DOF names derived from MJCF.

        Returns
        -------
        DOFAuditReport
            Detailed mismatch report.
        """
        data_names = motion.dof_names or []
        model_names = self._model_dof_names(robot_model)

        matched: list[str] = []
        mismatched: list[tuple[str, str]] = []
        unmatched_in_data: list[str] = []
        unmatched_in_model: list[str] = []

        max_len = max(len(data_names), len(model_names))
        for i in range(max_len):
            d = data_names[i] if i < len(data_names) else None
            m = model_names[i] if i < len(model_names) else None
            if d is None:
                if m is not None:
                    unmatched_in_model.append(m)
            elif m is None:
                unmatched_in_data.append(d)
            elif d == m:
                matched.append(d)
            else:
                mismatched.append((d, m))

        is_compatible = (
            len(mismatched) == 0
            and len(unmatched_in_data) == 0
            and len(unmatched_in_model) == 0
        )

        return DOFAuditReport(
            matched=matched,
            mismatched=mismatched,
            unmatched_in_data=unmatched_in_data,
            unmatched_in_model=unmatched_in_model,
            is_order_compatible=is_compatible,
        )

    def generate_sidecar(
        self,
        source_pkl_path: str | Path,
        robot_xml_path: str | Path,
        robot_name: str = "",
        source_pipeline: str = "",
        gmr_ik_config: str = "",
    ) -> dict:
        """Infer and return sidecar YAML content for *source_pkl_path*.

        The DOF names are taken from the MJCF joint order (as produced by
        the GMR pipeline).  This sidecar can then be saved next to the
        motion file to unlock name-based DOF mapping.

        Parameters
        ----------
        source_pkl_path:
            Path to the standard motion ``.pkl`` file.
        robot_xml_path:
            Path to the robot MJCF whose joint order was used during GMR.
        robot_name, source_pipeline, gmr_ik_config:
            Optional metadata strings written into the sidecar.

        Returns
        -------
        dict
            Sidecar YAML content as a plain Python dict.
        """
        adapter = ModelAdapter()
        robot_model = adapter.load_mjcf(robot_xml_path)
        dof_names = self._model_dof_names(robot_model)

        return {
            "dof_names": dof_names,
            "robot": robot_name or Path(robot_xml_path).stem,
            "source_pipeline": source_pipeline,
            "gmr_ik_config": gmr_ik_config,
        }

    def repair(
        self,
        motion: StandardMotion,
        canonical_order: list[str],
    ) -> StandardMotion:
        """Reorder ``dof_pos`` and ``dof_vel`` columns to *canonical_order*.

        Parameters
        ----------
        motion:
            Source motion clip with ``dof_names`` populated.
        canonical_order:
            Desired DOF order (list of names).

        Returns
        -------
        StandardMotion
            A new ``StandardMotion`` with reordered DOF arrays.

        Raises
        ------
        ValueError
            If ``motion.dof_names`` is ``None`` or a name in
            *canonical_order* is not found in ``motion.dof_names``.
        """
        if motion.dof_names is None:
            raise ValueError(
                "Cannot repair DOF order: motion.dof_names is None. "
                "Load a sidecar YAML first."
            )

        old_order = motion.dof_names
        index_map: list[int] = []
        for name in canonical_order:
            if name not in old_order:
                raise ValueError(
                    f"DOF name {name!r} not found in motion.dof_names."
                )
            index_map.append(old_order.index(name))

        import dataclasses
        return dataclasses.replace(
            motion,
            dof_pos=motion.dof_pos[:, index_map].copy(),
            dof_vel=motion.dof_vel[:, index_map].copy(),
            dof_names=canonical_order,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_dof_names(robot_model: RobotModel) -> list[str]:
        """Return DOF names from the model (based on its MJCF order)."""
        try:
            import mujoco as mj

            model = mj.MjModel.from_xml_path(robot_model.mjcf_path)
            names = []
            for jnt_id in range(model.njnt):
                if model.jnt_type[jnt_id] == mj.mjtJoint.mjJNT_FREE:
                    continue
                jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id) or ""
                names.append(jname)
            return names
        except Exception:  # noqa: BLE001
            return []
