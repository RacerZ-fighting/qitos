"""Replay/debug exports.

.. deprecated::
    ``qitos.debug`` is deprecated and will be removed in a future release.
    Use ``qitos.trace`` and ``qitos.qita`` for run inspection and replay.
"""

import warnings

warnings.warn(
    "qitos.debug is deprecated and will be removed in a future release. "
    "Use qitos.trace and qitos.qita for run inspection and replay.",
    DeprecationWarning,
    stacklevel=2,
)

from .breakpoints import Breakpoint
from .inspector import InspectorPayload, build_inspector_payload, compare_steps
from .replay import ReplaySession, ReplaySnapshot

__all__ = [
    "Breakpoint",
    "InspectorPayload",
    "build_inspector_payload",
    "compare_steps",
    "ReplaySession",
    "ReplaySnapshot",
]
