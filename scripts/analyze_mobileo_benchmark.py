#!/usr/bin/env python3
"""Summarize Mobile-O benchmark CSV (non-warmup runs). Usage: python3 analyze_mobileo_benchmark.py path/to/mobileo_benchmark_*.csv"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_mobileo_benchmark.py <benchmark.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    rows = list(csv.DictReader(path.open()))

    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("is_warmup", "").lower() == "true":
            continue
        key = (row["task"], row["phase"], row["metric"])
        try:
            groups[key].append(float(row["value"]))
        except ValueError:
            pass

    print(f"File: {path}")
    print(f"{'task':<20} {'phase':<16} {'metric':<28} {'mean':>10} {'min':>10} {'max':>10} n")
    print("-" * 100)
    for (task, phase, metric), values in sorted(groups.items()):
        if not values:
            continue
        mean = sum(values) / len(values)
        print(
            f"{task:<20} {phase:<16} {metric:<28} {mean:10.3f} {min(values):10.3f} {max(values):10.3f} {len(values)}"
        )


if __name__ == "__main__":
    main()
