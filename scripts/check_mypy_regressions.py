#!/usr/bin/env python3
"""Fail when a change introduces new mypy diagnostics vs. its base revision.

The comparison intentionally ignores line and column numbers so that moving
existing code does not turn unchanged diagnostics into regressions. It keeps
path, error code, and message, including duplicate counts.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

_ERROR = re.compile(
    r"^(?P<path>.+?\.py):\d+: error: (?P<message>.*?)(?: \[(?P<code>[^\]]+)\])?$"
)


def diagnostics(path: Path) -> Counter[tuple[str, str, str]]:
    found: Counter[tuple[str, str, str]] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ERROR.match(line)
        if match is None:
            continue
        key = (
            Path(match.group("path")).as_posix(),
            match.group("code") or "uncoded",
            match.group("message"),
        )
        found[key] += 1
    return found


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_mypy_regressions.py BASE.txt CURRENT.txt", file=sys.stderr)
        return 2
    baseline = diagnostics(Path(sys.argv[1]))
    current = diagnostics(Path(sys.argv[2]))
    regressions = current - baseline
    print(
        f"mypy diagnostics: base={sum(baseline.values())}, "
        f"current={sum(current.values())}, new={sum(regressions.values())}"
    )
    if not regressions:
        print("No new mypy diagnostics.")
        return 0
    for (path, code, message), count in sorted(regressions.items()):
        print(f"{path}: [{code}] {message} (new occurrences: {count})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
