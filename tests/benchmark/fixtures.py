"""
Synthetic test fixtures for the GMR benchmark test suite.

All fixtures generate data **without** requiring real LaFan1/AMASS datasets,
MuJoCo assets, or SMPL-X body models, so the test suite runs offline with
no external dependencies beyond ``numpy``.

Public helpers
--------------
make_synthetic_frames(n_frames, joint_names, seed)
    Generate a list of per-frame dicts in the format returned by
    ``load_bvh_file``: ``{joint_name: (pos_3d, quat_4d)}``.

make_synthetic_sequence(n_frames, joint_names, human_height, seed)
    Wrap ``make_synthetic_frames`` into the ``(frames, human_height)``
    tuple expected by ``RetargetingEvaluator.evaluate``.

make_synthetic_qpos(n_frames, nq, seed)
    Generate a random ``(n_frames, nq)`` qpos array.  Uses ``nq=36`` by
    default (7 free-joint DOFs + 29 G1 articulated DOFs).

LAFAN1_JOINT_NAMES
    Tuple of joint names matching those in ``bvh_lafan1_to_g1.json``.

make_lafan1_ik_config()
    Return a deep copy of the canonical bvh_lafan1_to_g1 config dict.
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Joint names present in the bvh_lafan1_to_g1.json human_scale_table and IK tables.
# These are the names the test-data generator must use so that IKConfigParamSpace
# can find them in the config.
LAFAN1_JOINT_NAMES: Tuple[str, ...] = (
    "Hips",
    "Spine2",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFootMod",
    "RightFootMod",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)

# Path to the canonical bvh_lafan1_to_g1.json inside the repository
_IK_CONFIG_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "general_motion_retargeting"
    / "ik_configs"
    / "bvh_lafan1_to_g1.json"
)

# ---------------------------------------------------------------------------
# Frame / sequence generators
# ---------------------------------------------------------------------------


def make_synthetic_frames(
    n_frames: int = 30,
    joint_names: Tuple[str, ...] = LAFAN1_JOINT_NAMES,
    seed: int = 0,
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Generate a list of synthetic per-frame dicts.

    Each frame is a dict ``{joint_name: (position, quaternion)}`` where:

    * ``position`` is a shape ``(3,)`` float64 array following a smooth
      sinusoidal trajectory so that consecutive frames are similar.
    * ``quaternion`` is a shape ``(4,)`` float64 unit quaternion.
      All orientations are set to the identity quaternion ``[1, 0, 0, 0]``.

    Parameters
    ----------
    n_frames : int
        Number of frames to generate.
    joint_names : tuple of str
        Joint names to include in each frame dict.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Length-``n_frames`` list of per-frame joint dicts.
    """
    rng = np.random.default_rng(seed)
    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []

    # Pre-generate random phase offsets for each joint so trajectories differ
    phases = rng.uniform(0, 2 * np.pi, size=(len(joint_names), 3))
    amplitudes = rng.uniform(0.05, 0.5, size=(len(joint_names), 3))

    # Identity quaternion (w, x, y, z) for all joints and frames
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for t in range(n_frames):
        # Normalised time in [0, 2π]
        t_norm = 2 * np.pi * t / max(n_frames - 1, 1)
        frame: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for j_idx, name in enumerate(joint_names):
            # Smooth sinusoidal position trajectory for each axis
            pos = amplitudes[j_idx] * np.sin(t_norm + phases[j_idx])
            frame[name] = (pos.astype(np.float64), identity_quat.copy())
        frames.append(frame)

    return frames


def make_synthetic_sequence(
    n_frames: int = 30,
    joint_names: Tuple[str, ...] = LAFAN1_JOINT_NAMES,
    human_height: float = 1.75,
    seed: int = 0,
) -> Tuple[List[Dict[str, Tuple[np.ndarray, np.ndarray]]], float]:
    """Create a ``(frames, human_height)`` tuple suitable for the evaluator.

    Parameters
    ----------
    n_frames : int
        Number of frames in the sequence.
    joint_names : tuple of str
        Joint names to include in each frame dict.
    human_height : float
        Assumed human height in metres.
    seed : int
        Random seed forwarded to :func:`make_synthetic_frames`.

    Returns
    -------
    tuple
        ``(frames, human_height)`` where ``frames`` is a list of per-frame
        dicts and ``human_height`` is the provided float.
    """
    frames = make_synthetic_frames(n_frames=n_frames, joint_names=joint_names, seed=seed)
    return frames, float(human_height)


def make_synthetic_qpos(
    n_frames: int = 30,
    nq: int = 36,
    seed: int = 0,
) -> np.ndarray:
    """Generate a random qpos array of shape ``(n_frames, nq)``.

    The default ``nq=36`` matches the Unitree G1 robot:
    7 free-joint DOFs (3 translation + 4 quaternion) + 29 articulated DOFs.
    Values are drawn uniformly from ``[-0.5, 0.5]`` so they are plausible
    but not necessarily within joint limits.

    Parameters
    ----------
    n_frames : int
        Number of time steps.
    nq : int
        Total number of qpos coordinates (free joint + articulated joints).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n_frames, nq)`` float64 array.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(n_frames, nq)).astype(np.float64)


# ---------------------------------------------------------------------------
# IK config helpers
# ---------------------------------------------------------------------------


def make_lafan1_ik_config() -> dict:
    """Return a deep copy of the canonical ``bvh_lafan1_to_g1`` IK config.

    Loads the JSON from the repository and returns an independent deep copy
    so that tests can modify it freely without affecting other tests or the
    on-disk file.

    Returns
    -------
    dict
        Complete IK config dict with all standard fields.

    Raises
    ------
    FileNotFoundError
        If ``bvh_lafan1_to_g1.json`` cannot be found in the expected location.
    """
    with open(_IK_CONFIG_PATH) as fh:
        return copy.deepcopy(json.load(fh))
