"""Compare initial-world builder timings and emit a Markdown summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        raise ValueError(f"invalid initial-world benchmark report: {path}")
    for row in result["results"]:
        if not row.get("partition_valid"):
            raise ValueError(f"invalid plate partition in benchmark report: {path}")
    return result


def _compare(baseline: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = {row["num_cells"]: row for row in baseline["results"]}
    head_rows = {row["num_cells"]: row for row in head["results"]}
    shared_sizes = sorted(set(baseline_rows) & set(head_rows))
    if not shared_sizes:
        raise ValueError("baseline and head reports have no matching mesh sizes")

    rows = []
    for size in shared_sizes:
        before = float(baseline_rows[size]["median_build_seconds"])
        after = float(head_rows[size]["median_build_seconds"])
        rows.append(
            {
                "num_cells": size,
                "baseline_seconds": before,
                "head_seconds": after,
                "change_percent": (after / before - 1.0) * 100.0,
                "speedup": before / after,
                "baseline_attempts": baseline_rows[size]["layout_attempts"],
                "head_attempts": head_rows[size]["layout_attempts"],
            }
        )
    return {
        "baseline_label": baseline.get("label", "baseline"),
        "head_label": head.get("label", "head"),
        "seed": baseline.get("seed"),
        "comparisons": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "### Initial-world construction benchmark",
        "",
        "Same runner; seed {} ; baseline {} ; head {}.".format(
            report["seed"], report["baseline_label"], report["head_label"]
        ),
        "",
        "| Cells | Baseline median | PR-head median | Change | Speedup |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        lines.append(
            "| {:,} | {:.3f}s | {:.3f}s | {:+.1f}% | {:.2f}x |".format(
                row["num_cells"],
                row["baseline_seconds"],
                row["head_seconds"],
                row["change_percent"],
                row["speedup"],
            )
        )
    lines.extend(
        [
            "",
            "Each builder timing is measured after the configured warm-up; mesh "
            "construction is reported separately. Plate-partition correctness is "
            "checked for every timed and warm-up build. Two measured repetitions "
            "are used by default; interpret small changes with runner noise in mind.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--output", default="initial-world-comparison.json")
    args = parser.parse_args()

    report = _compare(_load(args.baseline), _load(args.head))
    markdown = _markdown(report)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(markdown)
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as summary:
            summary.write(markdown)


if __name__ == "__main__":
    main()
