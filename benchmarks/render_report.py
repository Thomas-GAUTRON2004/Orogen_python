"""Render the paired benchmark JSON into Markdown and CSV reports."""

import argparse
import csv
import json
from pathlib import Path

def core_labels(graph_cells: int) -> dict[str, str]:
    """Build workload labels from the actual benchmark configuration."""
    cells = f"{graph_cells:,}"
    return {
        "noise_fbm_ms": "3-D simplex FBM (40k points x 5 octaves)",
        "spherical_geometry_ms": "Spherical geometry (80k points)",
        "mesh_bfs_ms": f"Mesh BFS ({cells}-cell degree-6 ring)",
        "mesh_components_ms": f"Connected components ({cells}-cell graph)",
        "mesh_dijkstra_ms": f"Mesh Dijkstra ({cells}-cell ring, 20,000 km cutoff)",
        "crust_create_snapshot_ms": "Crust create + batch snapshot (6k cells)",
        "crust_copy_ms": "Crust batch copy (6k cells)",
    }


def fmt(value, digits: int = 3) -> str:
    """Format a numeric value in milliseconds or return an unavailable marker."""
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def get_median(section, key: str):
    """Read one median from a nested timing report."""
    if not section:
        return None
    return section.get(key, {}).get("median_ms")


def ratio(numerator, denominator):
    """Return a speed ratio, with values above one meaning the numerator wins."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def timing_table(report: dict) -> list[str]:
    """Build the paired JavaScript/Python microbenchmark comparison."""
    lines = [
        "| Operation | JavaScript | Python pure | Python + Numba | Pure / JS | Numba / JS | Numba gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    pure = report["python"]["pure"]["timings"]
    accelerated = report["python"].get("numba")
    labels = core_labels(report["configuration"]["graph_cells"])
    for key, label in labels.items():
        js_ms = get_median(report["javascript"], key)
        pure_ms = get_median(pure, key)
        numba_ms = get_median(accelerated["timings"], key) if accelerated else None
        pure_ratio = ratio(pure_ms, js_ms)
        numba_ratio = ratio(numba_ms, js_ms)
        gain = ratio(pure_ms, numba_ms)
        lines.append(
            f"| {label} | {fmt(js_ms)} | {fmt(pure_ms)} | {fmt(numba_ms)} | "
            f"{fmt(pure_ratio, 2) + 'x' if pure_ratio is not None else 'n/a'} | "
            f"{fmt(numba_ratio, 2) + 'x' if numba_ratio is not None else 'n/a'} | "
            f"{fmt(gain, 2) + 'x' if gain is not None else 'n/a'} |"
        )
    return lines


def python_only_table(report: dict) -> list[str]:
    """Build the Python-only tectonic and far-field performance table."""
    lines = [
        "| Operation | Python pure | Python + Numba | Numba speedup |",
        "|---|---:|---:|---:|",
    ]
    pure = report["python"]["pure"]
    accelerated = report["python"].get("numba")
    rows = [
        (
            (
                "Far-field spread_push "
                f"({report['configuration']['graph_cells']:,} cells, two sources)"
            ),
            pure.get("far_field_diffusion"),
            accelerated.get("far_field_diffusion") if accelerated else None,
        ),
        (
            "One step_world (spherical mesh, four plates)",
            pure.get("step_world"),
            accelerated.get("step_world") if accelerated else None,
        ),
    ]
    for label, pure_stats, numba_stats in rows:
        pure_ms = pure_stats.get("median_ms") if pure_stats else None
        numba_ms = numba_stats.get("median_ms") if numba_stats else None
        gain = ratio(pure_ms, numba_ms)
        lines.append(
            f"| {label} | {fmt(pure_ms)} | {fmt(numba_ms)} | "
            f"{fmt(gain, 2) + 'x' if gain is not None else 'n/a'} |"
        )
    return lines


def runtime_lines(report: dict) -> list[str]:
    """Describe the runtime and CPU environment associated with a result."""
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


def csv_rows(report: dict) -> list[dict]:
    """Flatten every timing sample set into rows for spreadsheet analysis."""
    rows = []
    labels = core_labels(report["configuration"]["graph_cells"])
    for key, label in labels.items():
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
        for runtime_name, section in sections:
            stats = section.get(key) if section else None
            if stats:
                rows.append({"operation": label, "runtime": runtime_name, **stats})

    for key, label in (
        (
            "far_field_diffusion",
            f"Far-field spread_push ({report['configuration']['graph_cells']:,} cells)",
        ),
        ("step_world", "step_world (four plates)"),
    ):
        sections = (
            ("Python pure", report["python"]["pure"]),
            ("Python + Numba", report["python"].get("numba")),
        )
        for runtime_name, section in sections:
            stats = section.get(key) if section else None
            if stats:
                rows.append({"operation": label, "runtime": runtime_name, **stats})
    return rows


def main() -> None:
    """Generate Markdown, CSV, and an optional GitHub Actions step summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--markdown", type=Path, default=Path("benchmark-results.md"))
    parser.add_argument("--csv", type=Path, default=Path("benchmark-results.csv"))
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    report = json.loads(args.json_file.read_text(encoding="utf-8"))
    configuration = report["configuration"]
    lines = [
        "# JavaScript / Python / Numba benchmark",
        "",
        f"- UTC date: {report['generated_at']}",
        f"- Samples: median of {configuration['rounds']} measurements "
        f"after {configuration['warmups']} warmups",
        "",
        "## Common workloads",
        "",
        *timing_table(report),
        "",
        "The Python / JS ratio is Python time divided by JavaScript time; "
        "a value below 1 means Python is faster. Numba gain is Python pure "
        "time divided by Python + Numba time.",
        "",
        "## Python-only tectonic workloads",
        "",
        *python_only_table(report),
        "",
        "step_world has no JavaScript row: this repository does not yet contain "
        "a complete equivalent JavaScript tectonic orchestrator. The measured "
        "time covers one step of the current Python implementation on a "
        "prebuilt four-plate world; mesh and world construction are excluded.",
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
        "- BFS, connected components, and Dijkstra use a synthetic degree-6 CSR "
        "ring graph. Dijkstra stops at a 20,000 km cutoff.",
        "- FBM and geometry use batch APIs in Python and scalar reference loops "
        "in JavaScript; the measurements compare the execution paths exposed "
        "by each implementation.",
        "- Crust creation/snapshot and copying compare equivalent logical work, "
        "but Python batch APIs do not reproduce JavaScript scalar call granularity.",
        "- Python process startup, sphere-mesh creation, and step_world world "
        "preparation are excluded from the timed regions.",
        "",
    ]
    markdown = "\n".join(lines)
    args.markdown.write_text(markdown, encoding="utf-8")

    rows = csv_rows(report)
    columns = [
        "operation",
        "runtime",
        "median_ms",
        "mean_ms",
        "stddev_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "samples_ms",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**row, "samples_ms": json.dumps(row.get("samples_ms", []))}
            )

    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(markdown)
            handle.write("\n")

    print(f"Wrote {args.markdown} and {args.csv}")


if __name__ == "__main__":
    main()
