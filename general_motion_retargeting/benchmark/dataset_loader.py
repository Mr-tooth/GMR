"""
Dataset Loader for GMR Benchmark.

Supports loading and sampling motion sequences from:

* **LaFan1** (BVH format): uses the existing
  ``general_motion_retargeting.utils.lafan1.load_bvh_file`` helper.
* **AMASS** (SMPL-X ``.npz`` format): uses
  ``general_motion_retargeting.utils.smpl.load_smplx_file`` /
  ``get_smplx_data_offline_fast``.

Usage example
-------------
::

    loader = DatasetLoader(
        dataset_type="lafan1",
        data_path="/data/lafan1/",
        max_sequences=10,
        max_frames_per_seq=200,
    )
    sequences = loader.load_sequences()
    # sequences: list of (frames, human_height) tuples where
    #   frames is List[dict] – one dict per frame mapping joint name → (pos, quat)
    #   human_height is float

Notes
-----
For optimisation runs, keeping ``max_frames_per_seq`` small (e.g. 200) greatly
speeds up each Optuna trial without sacrificing metric quality.
"""

from __future__ import annotations

import pathlib
import random
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Frame = dict  # {joint_name: (np.ndarray[3], np.ndarray[4])}
Sequence = Tuple[List[Frame], float]  # (frames, human_height)


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Load motion sequences from disk for use in benchmark evaluation.

    Parameters
    ----------
    dataset_type:
        ``'lafan1'`` – BVH files from the LaFan1 dataset.
        ``'amass'``  – ``.npz`` SMPL-X files from the AMASS dataset.
    data_path:
        Root directory containing the dataset files.  For LaFan1 this should
        point at the directory holding ``.bvh`` files (or a parent thereof).
        For AMASS it should point at the directory holding ``.npz`` files.
    max_sequences:
        Maximum number of sequences to load.  ``None`` means load all found
        files.
    max_frames_per_seq:
        Maximum number of frames to keep from each sequence.  Frames are
        taken from the *beginning* of each sequence.  ``None`` means keep
        all frames.
    target_fps:
        Target frame rate for AMASS data (SMPL-X source fps may differ).
        Ignored for LaFan1.
    smplx_body_model_path:
        Path to the SMPL-X body model directory (required for AMASS).
        Typically ``<repo_root>/assets/body_models``.
    seed:
        Random seed used when shuffling found files before truncation.
    verbose:
        Print progress messages if ``True``.
    """

    def __init__(
        self,
        dataset_type: str,
        data_path: str | pathlib.Path,
        max_sequences: Optional[int] = None,
        max_frames_per_seq: Optional[int] = 200,
        target_fps: int = 30,
        smplx_body_model_path: Optional[str | pathlib.Path] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        if dataset_type not in ("lafan1", "amass"):
            raise ValueError(f"dataset_type must be 'lafan1' or 'amass', got {dataset_type!r}")

        self.dataset_type = dataset_type
        self.data_path = pathlib.Path(data_path)
        self.max_sequences = max_sequences
        self.max_frames_per_seq = max_frames_per_seq
        self.target_fps = target_fps
        self.smplx_body_model_path = (
            pathlib.Path(smplx_body_model_path) if smplx_body_model_path else None
        )
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_sequences(self) -> List[Sequence]:
        """Discover and load all (or up to *max_sequences*) sequences.

        Returns
        -------
        List[Sequence]
            Each element is ``(frames, human_height)`` where ``frames`` is a
            list of per-frame dicts mapping joint name to ``(position, quaternion)``.
        """
        if self.dataset_type == "lafan1":
            return self._load_lafan1()
        else:
            return self._load_amass()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_files(self, pattern: str) -> List[pathlib.Path]:
        """Recursively find all files matching *pattern* under ``data_path``."""
        files = sorted(self.data_path.rglob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found under {self.data_path}"
            )
        rng = random.Random(self.seed)
        rng.shuffle(files)
        if self.max_sequences is not None:
            files = files[: self.max_sequences]
        return files

    def _truncate(self, frames: List[Frame]) -> List[Frame]:
        if self.max_frames_per_seq is not None:
            return frames[: self.max_frames_per_seq]
        return frames

    def _load_lafan1(self) -> List[Sequence]:
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        files = self._find_files("*.bvh")
        if self.verbose:
            print(f"[DatasetLoader] Found {len(files)} BVH file(s) under {self.data_path}")

        sequences: List[Sequence] = []
        for bvh_path in files:
            try:
                frames, human_height = load_bvh_file(str(bvh_path), format="lafan1")
                frames = self._truncate(frames)
                sequences.append((frames, human_height))
                if self.verbose:
                    print(f"  Loaded {len(frames)} frames from {bvh_path.name}")
            except Exception as exc:
                if self.verbose:
                    print(f"  [WARN] Failed to load {bvh_path.name}: {exc}")
        return sequences

    def _load_amass(self) -> List[Sequence]:
        from general_motion_retargeting.utils.smpl import (
            load_smplx_file,
            get_smplx_data_offline_fast,
        )

        if self.smplx_body_model_path is None:
            raise ValueError(
                "smplx_body_model_path must be set when dataset_type='amass'"
            )

        files = self._find_files("*.npz")
        if self.verbose:
            print(f"[DatasetLoader] Found {len(files)} NPZ file(s) under {self.data_path}")

        sequences: List[Sequence] = []
        for npz_path in files:
            try:
                smplx_data, body_model, smplx_output, human_height = load_smplx_file(
                    str(npz_path), str(self.smplx_body_model_path)
                )
                frames, _ = get_smplx_data_offline_fast(
                    smplx_data, body_model, smplx_output, tgt_fps=self.target_fps
                )
                frames = self._truncate(frames)
                sequences.append((frames, human_height))
                if self.verbose:
                    print(f"  Loaded {len(frames)} frames from {npz_path.name}")
            except Exception as exc:
                if self.verbose:
                    print(f"  [WARN] Failed to load {npz_path.name}: {exc}")
        return sequences
