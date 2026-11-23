"""Render benchmark JSON using the declarative workload catalog.

To expose another workload in reports, register its stable ID in
benchmarks/catalog.json and emit timings under that ID from the language runners.
"""

import argparse
import csv
import json
from pathlib import Path


def load_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1:
        raise ValueError(f"unsupported benchmark catalog schema: {catalog.get('schema_version')}")
    for group in ("paired", "python_only"):
        ids = [item["id"] for item in catalog.get(group, [])]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate benchmark IDs in catalog group {group}")
    return catalog


def validate_report(report: dict, catalog: dict) -> None:
    """Fail clearly when a catalogued workload is missing from a runner."""
    paired = catalog.get("paired", [])
    javascript = report.get("javascript", {})
    pure = report.get("python", {}).get("pure", {}).get("timings", {})
    missing_js = [item["id"] for item in paired if item["id"] not in javascript]
    python_only = catalog.get("python_only", [])
    required_python = paired + python_only
    pure_root = report.get("python", {}).get("pure", {})
    missing_pure = [
        item["id"] for item in required_python
        if item["id"] not in (
            pure_root.get("timings", {}) if item in paired else pure_root
        )
    ]
    missing_numba = []
    accelerated = report.get("python", {}).get("numba")
    if accelerated is not None:
        missing_numba = [
            item["id"] for item in required_python
            if item["id"] not in (
                accelerated.get("timings", {}) if item in paired else accelerated
            )
        ]
    if missing_js or missing_pure or missing_numba:
        raise ValueError(
            "benchmark catalog/runner mismatch: "
            f"missing JavaScript workloads={missing_js}; "
            f"missing Python pure workloads={missing_pure}; "
            f"missing Python + Numba workloads={missing_numba}"
        )


def fmt(value, digits: int = 3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def get_median(section, key: str):
    if not section:
        return None
    return section.get(key, {}).get("median_ms")


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _ratio_text(value):
    return f"{fmt(value, 2)}x" if value is not None else "n/a"


def timing_table(report: dict, catalog: dict) -> list[str]:
    lines = [
        "| Operation | JavaScript | Python pure | Python + Numba | Pure / JS | Numba / JS | Numba gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    pure = report["python"]["pure"]["timings"]
    accelerated = report["python"].get("numba")
    for item in catalog.get("paired", []):
        key = item["id"]
        js_ms = get_median(report["javascript"], key)
        pure_ms = get_median(pure, key)
        numba_ms = get_median(accelerated["timings"], key) if accelerated else None
        lines.append(
            f"| {item['label']} | {fmt(js_ms)} | {fmt(pure_ms)} | {fmt(numba_ms)} | "
            f"{_ratio_text(ratio(pure_ms, js_ms))} | "
            f"{_ratio_text(ratio(numba_ms, js_ms))} | "
            f"{_ratio_text(ratio(pure_ms, numba_ms))} |"
        )
    if len(lines) == 2:
        lines.append("| No paired workloads registered | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def python_only_table(report: dict, catalog: dict) -> list[str]:
    lines = [
        "| Operation | Python pure | Python + Numba | Numba speedup |",
        "|---|---:|---:|---:|",
    ]
    pure = report["python"]["pure"]
    accelerated = report["python"].get("numba")
    for item in catalog.get("python_only", []):
        key = item["id"]
        pure_stats = pure.get(key)
        numba_stats = accelerated.get(key) if accelerated else None
        pure_ms = pure_stats.get("median_ms") if pure_stats else None
        numba_ms = numba_stats.get("median_ms") if numba_stats else None
        lines.append(
            f"| {item['label']} | {fmt(pure_ms)} | {fmt(numba_ms)} | "
            f"{_ratio_text(ratio(pure_ms, numba_ms))} |"
        )
    if len(lines) == 2:
        lines.append("| No Python-only workloads registered | n/a | n/a | n/a |")
    return lines


def runtime_lines(report: dict) -> list[str]:
    runtime = report["runtime"]
    python = runtime["python"]
    cpu = runtime.get("cpu_model") or python.get("processor") or "not reported"
    return [
        f"- JavaScript: Node {runtime['node']}",
        f"- Python: {python['python']} ({python['implementation']}), NumPy {python['numpy']}",
        f"- Numba available: {'yes' if report['python']['numba_available'] else 'no'}",
        f"- System: {runtime['platform']} / {runtime['arch']}",
        f"- Processor: {cpu}",
        f"- Measured repetitions: {report['configuration']['rounds']}",
        f"- Excluded warmups: {report['configuration']['warmups']}",
    ]


def csv_rows(report: dict, catalog: dict) -> list[dict]:
    rows = []
    sections = (
        ("JavaScript", report["javascript"]),
        ("Python pure", report["python"]["pure"]["timings"]),
        (
            "Python + Numba",
            report["python"]["numba"]["timings"]
            if report["python"].get("numba")
            else None,
        ),
    )
    for item in catalog.get("paired", []):
        for runtime_name, section in sections:
            stats = section.get(item["id"]) if section else None
            if stats:
                rows.append({"operation": item["label"], "runtime": runtime_name, **stats})

    for item in catalog.get("python_only", []):
        for runtime_name, section in (
            ("Python pure", report["python"]["pure"]),
            ("Python + Numba", report["python"].get("numba")),
        ):
            stats = section.get(item["id"]) if section else None
            if stats:
                rows.append({"operation": item["label"], "runtime": runtime_name, **stats})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--registry", type=Path, default=Path(__file__).with_name("catalog.json"))
    parser.add_argument("--markdown", type=Path, default=Path("benchmark-results.md"))
    parser.add_argument("--csv", type=Path, default=Path("benchmark-results.csv"))
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    report = json.loads(args.json_file.read_text(encoding="utf-8"))
    catalog = load_catalog(args.registry)
    validate_report(report, catalog)
    configuration = report["configuration"]
    lines = [
        "# JavaScript / Python / Numba benchmark",
        "",
        f"- UTC date: {report['generated_at']}",
        f"- Samples: median of {configuration['rounds']} measurements "
        f"after {configuration['warmups']} warmups",
        "",
        "## Paired workloads",
        "",
        *timing_table(report, catalog),
        "",
        "The Python / JS ratio is Python time divided by JavaScript time; "
        "a value below 1 means Python is faster. Numba gain is Python pure "
        "time divided by Python + Numba time.",
        "",
        "## Python-only workloads",
        "",
        *python_only_table(report, catalog),
        "",
        "A Python-only entry has no JavaScript equivalent registered yet. "
        "It remains directly comparable between the Python fallback and Numba.",
        "",
        "## Environment",
        "",
        *runtime_lines(report),
        "",
        "## Statistics",
        "",
        "The JSON preserves raw samples, median, mean, sample standard deviation, "
        "95th percentile, minimum, and maximum for every measurement. Times are "
        "reported in milliseconds.",
        "",
        "## Interpretation limits",
        "",
        "- Timings come from a shared GitHub-hosted runner and are specific to "
        "that runner and revision, not universal guarantees.",
        "- Each workload's fixture and timed boundary are defined alongside its "
        "registration in the benchmark runners.",
        "- Python process startup and fixture preparation should remain outside "
        "timed regions unless they are explicitly part of the operation being measured.",
        "- Numba compilation warmup is excluded from measured samples.",
        "",
    ]
    markdown = "\n".join(lines)
    args.markdown.write_text(markdown, encoding="utf-8")

    rows = csv_rows(report, catalog)
    columns = [
        "operation", "runtime", "median_ms", "mean_ms", "stddev_ms",
        "p95_ms", "min_ms", "max_ms", "samples_ms",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "samples_ms": json.dumps(row.get("samples_ms", []))})

    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
    print(f"Wrote {args.markdown} and {args.csv}")


if __name__ == "__main__":
    main()
