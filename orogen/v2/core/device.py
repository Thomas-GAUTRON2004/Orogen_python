"""Device memory-budget classification from V2 core/device.js.

Python has no browser navigator or matchMedia globals. The module-level
constant therefore follows the upstream Node/server behavior (False); callers
that have browser capability data can use is_small_device explicitly.
"""
from __future__ import annotations

SMALL_DEVICE = False


def is_small_device(device_memory_gb: float | None = None, coarse_pointer: bool = False) -> bool:
    """Apply the upstream rule: reported memory <= 4 GB or coarse pointer."""
    memory = 8.0 if device_memory_gb is None else float(device_memory_gb)
    return memory <= 4.0 or bool(coarse_pointer)
