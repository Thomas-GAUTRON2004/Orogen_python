"""Shared harness for tests that compare the Python port with upstream JavaScript.

A differential test must build one input payload, execute the upstream reference with
that payload, execute the Python implementation with the same payload, then compare
the two outputs. Do not use this helper to compare either implementation with a
third, hand-written expected result.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_js_reference(
    runner: str | Path,
    payload: Any,
    *,
    timeout: int = 120,
) -> Any:
    """Run an upstream JavaScript reference runner with a JSON input payload."""
    node = shutil.which("node")
    if node is None:
        message = "Node.js is required for JavaScript/Python differential tests"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)

    runner_path = Path(runner)
    if not runner_path.is_absolute():
        runner_path = ROOT / runner_path
    request = json.dumps(_to_json_value(payload), allow_nan=False, separators=(",", ":"))
    result = subprocess.run(
        [node, str(runner_path)],
        input=request,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode:
        pytest.fail(
            f"JavaScript reference runner failed ({result.returncode}):\n"
            f"{result.stderr}\n{result.stdout}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        pytest.fail(f"JavaScript reference emitted invalid JSON: {error}\n{result.stdout}")


def assert_js_python_equal(
    python_value: Any,
    javascript_value: Any,
    *,
    rtol: float = 2e-7,
    atol: float = 2e-9,
    path: str = "$",
) -> None:
    """Compare Python output recursively with output produced by JavaScript."""
    left = _to_json_value(python_value)
    right = _to_json_value(javascript_value)

    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys(), (
            f"{path}: object keys differ: Python={sorted(left)}, "
            f"JavaScript={sorted(right)}"
        )
        for key in left:
            assert_js_python_equal(
                left[key], right[key], rtol=rtol, atol=atol, path=f"{path}.{key}"
            )
        return

    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), (
            f"{path}: array lengths differ: Python={len(left)}, "
            f"JavaScript={len(right)}"
        )
        for index, (py_item, js_item) in enumerate(zip(left, right, strict=True)):
            assert_js_python_equal(
                py_item, js_item, rtol=rtol, atol=atol, path=f"{path}[{index}]"
            )
        return

    if isinstance(left, bool) or isinstance(right, bool):
        assert left is right, f"{path}: Python={left!r}, JavaScript={right!r}"
        return

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if math.isinf(left) or math.isinf(right) or math.isnan(left) or math.isnan(right):
            assert left == right, f"{path}: Python={left!r}, JavaScript={right!r}"
        else:
            assert math.isclose(left, right, rel_tol=rtol, abs_tol=atol), (
                f"{path}: Python={left!r}, JavaScript={right!r}"
            )
        return

    assert left == right, f"{path}: Python={left!r}, JavaScript={right!r}"


def _to_json_value(value: Any) -> Any:
    """Convert NumPy-backed Python values to the JSON data model."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return value
