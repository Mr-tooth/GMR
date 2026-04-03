"""
Root conftest.py – stub out heavy optional dependencies.

``general_motion_retargeting/__init__.py`` imports ``kinematics_model`` and
``neck_retarget`` which depend on ``torch`` and ``smplx`` that are not
available in the lightweight CI environment.  Because our benchmark tests
only need the ``benchmark`` sub-package (and the core ``motion_retarget``
module), we pre-install minimal stubs in ``sys.modules`` before any test
module is collected.

This conftest is deliberately placed at the *tests/* level so that it runs
before any import of ``general_motion_retargeting`` occurs.
"""

from __future__ import annotations

import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    """Return an empty module stub registered under *name*."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# Stub: torch (including jit.script passthrough, nn.Module, Tensor)
# ---------------------------------------------------------------------------
_torch = _make_stub("torch")
_torch.Tensor = object  # type: ignore[attr-defined]

# torch.jit – provide a passthrough @script decorator
_jit = _make_stub("torch.jit")
_jit.script = lambda fn: fn  # type: ignore[attr-defined]
_torch.jit = _jit  # type: ignore[attr-defined]

# torch.nn
_nn = _make_stub("torch.nn")
_nn.Module = object  # type: ignore[attr-defined]
_torch.nn = _nn  # type: ignore[attr-defined]
_make_stub("torch.nn.functional")

# torch math stubs (used in torch_utils.py)
import math as _math
_torch.atan2 = _math.atan2  # type: ignore[attr-defined]
_torch.asin = _math.asin  # type: ignore[attr-defined]
_torch.sin = _math.sin  # type: ignore[attr-defined]
_torch.cos = _math.cos  # type: ignore[attr-defined]
_torch.clip = lambda x, lo, hi: max(lo, min(hi, x))  # type: ignore[attr-defined]

_make_stub("torch.utils")
_make_stub("torch.utils.data")

# ---------------------------------------------------------------------------
# Stub: smplx
# ---------------------------------------------------------------------------
_smplx = _make_stub("smplx")
_smplx.create = lambda *a, **kw: None  # type: ignore[attr-defined]
_smplx.body_models = _make_stub("smplx.body_models")

# ---------------------------------------------------------------------------
# Stub: xrobotoolkit_sdk (optional C++ binding)
# ---------------------------------------------------------------------------
_make_stub("xrobotoolkit_sdk")
