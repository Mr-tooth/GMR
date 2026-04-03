"""
Tests for general_motion_retargeting.benchmark.dataset_loader.

Uses a temporary directory populated with minimal stub files so no real
LaFan1/AMASS data is required.  The ``_load_lafan1`` and ``_load_amass``
paths are tested via partial mocks to avoid the full BVH/SMPL-X pipeline.
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from general_motion_retargeting.benchmark.dataset_loader import DatasetLoader
from tests.benchmark.fixtures import make_synthetic_frames, LAFAN1_JOINT_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_bvh_files(tmpdir: pathlib.Path, n: int = 5) -> list[pathlib.Path]:
    """Create *n* empty ``.bvh`` stub files in *tmpdir*."""
    paths = []
    for i in range(n):
        p = tmpdir / f"seq_{i:03d}.bvh"
        p.write_text(f"stub bvh {i}")
        paths.append(p)
    return paths


def _patched_loader(
    tmpdir: pathlib.Path,
    max_sequences: int | None = None,
    max_frames: int | None = 200,
    seed: int = 42,
) -> DatasetLoader:
    """Return a DatasetLoader configured to read from *tmpdir*."""
    return DatasetLoader(
        dataset_type="lafan1",
        data_path=tmpdir,
        max_sequences=max_sequences,
        max_frames_per_seq=max_frames,
        seed=seed,
        verbose=False,
    )


# Patch target: load_bvh_file is imported locally inside _load_lafan1
_LOAD_BVH_TARGET = "general_motion_retargeting.utils.lafan1.load_bvh_file"


def _fake_load_bvh(path: str, format: str = "lafan1"):  # noqa: A002
    """Stub for load_bvh_file: returns 50 synthetic frames at height 1.75."""
    frames = make_synthetic_frames(n_frames=50, joint_names=LAFAN1_JOINT_NAMES)
    return frames, 1.75


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindFilesTruncation:
    """DatasetLoader must honour max_sequences."""

    def test_truncates_to_max_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            _make_stub_bvh_files(tmpdir, n=5)

            loader = _patched_loader(tmpdir, max_sequences=2)
            with patch(
                _LOAD_BVH_TARGET,
                side_effect=_fake_load_bvh,
            ):
                seqs = loader.load_sequences()

        assert len(seqs) == 2

    def test_no_truncation_returns_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            _make_stub_bvh_files(tmpdir, n=3)

            loader = _patched_loader(tmpdir, max_sequences=None)
            with patch(
                _LOAD_BVH_TARGET,
                side_effect=_fake_load_bvh,
            ):
                seqs = loader.load_sequences()

        assert len(seqs) == 3


class TestTruncateFrames:
    """Loaded sequences must be capped at max_frames_per_seq."""

    def test_frames_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            _make_stub_bvh_files(tmpdir, n=2)

            # Stub returns 50 frames; loader should cut to 20
            loader = _patched_loader(tmpdir, max_sequences=2, max_frames=20)
            with patch(
                _LOAD_BVH_TARGET,
                side_effect=_fake_load_bvh,
            ):
                seqs = loader.load_sequences()

        for frames, _ in seqs:
            assert len(frames) <= 20

    def test_no_frame_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            _make_stub_bvh_files(tmpdir, n=1)

            loader = _patched_loader(tmpdir, max_sequences=1, max_frames=None)
            with patch(
                _LOAD_BVH_TARGET,
                side_effect=_fake_load_bvh,
            ):
                seqs = loader.load_sequences()

        # Stub returns 50 frames; with no cap all 50 should be present
        assert len(seqs[0][0]) == 50


class TestFindFilesNotFound:
    """An empty directory must raise FileNotFoundError."""

    def test_empty_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = _patched_loader(pathlib.Path(tmpdir))
            with pytest.raises(FileNotFoundError):
                loader.load_sequences()


class TestInvalidDatasetType:
    """Constructing a DatasetLoader with an unknown type must raise ValueError."""

    def test_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="dataset_type must be"):
            DatasetLoader(
                dataset_type="unknown_format",
                data_path="/tmp",
            )


class TestSeedReproducibility:
    """The same seed must produce the same file order across two calls."""

    def test_same_seed_same_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            _make_stub_bvh_files(tmpdir, n=6)

            # Capture file order by recording paths passed to load_bvh_file
            recorded: list[list[str]] = []

            def _record_and_load(path: str, format: str = "lafan1"):  # noqa: A002
                recorded[-1].append(path)
                return _fake_load_bvh(path, format)

            for _ in range(2):
                recorded.append([])
                loader = _patched_loader(tmpdir, max_sequences=None, seed=99)
                with patch(
                    _LOAD_BVH_TARGET,
                    side_effect=_record_and_load,
                ):
                    loader.load_sequences()

        # Both runs must visit files in the same order
        assert recorded[0] == recorded[1]

    def test_different_seeds_may_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            _make_stub_bvh_files(tmpdir, n=6)

            orders: list[list[str]] = []

            for seed in (0, 1):
                order: list[str] = []

                def _record_and_load(path: str, format: str = "lafan1", _o=order):  # noqa: A002
                    _o.append(path)
                    return _fake_load_bvh(path, format)

                loader = _patched_loader(tmpdir, max_sequences=None, seed=seed)
                with patch(
                    _LOAD_BVH_TARGET,
                    side_effect=_record_and_load,
                ):
                    loader.load_sequences()
                orders.append(order)

        # Seeds 0 and 1 are verified to shuffle 6 files into different orderings.
        # This is a property of Python's random.Random shuffle on a specific input
        # rather than a probabilistic test, so it is deterministic.
        assert orders[0] != orders[1], (
            "Expected different seeds to produce different file orders "
            f"(seed=0: {orders[0]}, seed=1: {orders[1]})"
        )
