#!/usr/bin/env python3
"""Build the portable Recall@10 normalized-energy data for Figure 17."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from energy_model import (  # noqa: E402
    DATASET_ORDER,
    DEFAULT_FIGURE14_CSV,
    DEFAULT_TRACE_TABLE,
    build_energy_rows,
    geometric_mean,
    write_csv,
)


DEFAULT_OUTPUT = SCRIPT_DIR / "data/figure17_energy_breakdown.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "output/figure17_summary.tsv"
METHOD_ORDER = [
    "NMP-Base",
    "ANSMET",
    "NMP-FPMA",
    "NMP-FPSA",
    "NMP-Base-ET",
    "NMP-FPSA-ET",
    "MFNNS",
]
OUTPUT_FIELDS = [
    "top_k",
    "dataset_key",
    "dataset_label",
    "dataset_short",
    "method",
    "trace_design",
    "trace_source_design",
    "compute_formula",
    "compute_energy_nj",
    "memory_energy_nj",
    "total_energy_nj",
    "compute_energy_norm_to_nmp_base",
    "memory_energy_norm_to_nmp_base",
    "total_energy_norm_to_nmp_base",
    "config_ref",
    "trace_table_ref",
    "data_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-table", type=Path, default=DEFAULT_TRACE_TABLE)
    parser.add_argument("--figure14", type=Path, default=DEFAULT_FIGURE14_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def build_rows(trace_path: Path, figure14_path: Path) -> list[dict[str, object]]:
    energy_rows = build_energy_rows(trace_path, figure14_path)
    lookup = {
        (str(row["dataset_key"]), str(row["method"])): row
        for row in energy_rows
    }
    rows: list[dict[str, object]] = []
    for dataset_key, _, _ in DATASET_ORDER:
        baseline = float(lookup[(dataset_key, "NMP-Base")]["total_energy_nj"])
        for method in METHOD_ORDER:
            source = lookup[(dataset_key, method)]
            row = {field: source.get(field, "") for field in OUTPUT_FIELDS}
            row["compute_energy_norm_to_nmp_base"] = (
                float(source["compute_energy_nj"]) / baseline
            )
            row["memory_energy_norm_to_nmp_base"] = (
                float(source["memory_energy_nj"]) / baseline
            )
            row["total_energy_norm_to_nmp_base"] = (
                float(source["total_energy_nj"]) / baseline
            )
            rows.append(row)
    expected = len(DATASET_ORDER) * len(METHOD_ORDER)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} Figure 17 rows, found {len(rows)}")
    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lookup = {
        (str(row["dataset_key"]), str(row["method"])): float(
            row["total_energy_nj"]
        )
        for row in rows
    }
    lines = ["metric\tvalue"]
    comparisons = [
        ("MFNNS", "ANSMET"),
        ("MFNNS", "NMP-Base"),
        ("NMP-Base-ET", "NMP-Base"),
        ("NMP-FPSA-ET", "NMP-Base-ET"),
        ("MFNNS", "NMP-FPSA-ET"),
    ]
    for method, baseline in comparisons:
        ratios = [
            lookup[(dataset_key, method)] / lookup[(dataset_key, baseline)]
            for dataset_key, _, _ in DATASET_ORDER
        ]
        sum_ratio = sum(
            lookup[(dataset_key, method)] for dataset_key, _, _ in DATASET_ORDER
        ) / sum(
            lookup[(dataset_key, baseline)]
            for dataset_key, _, _ in DATASET_ORDER
        )
        prefix = f"{method}_over_{baseline}"
        lines.append(f"sum_energy_ratio:{prefix}\t{sum_ratio:.15f}")
        lines.append(f"sum_energy_reduction:{prefix}\t{1-sum_ratio:.15f}")
        gm = geometric_mean(ratios)
        lines.append(f"geomean_energy_ratio:{prefix}\t{gm:.15f}")
        lines.append(f"geomean_energy_reduction:{prefix}\t{1-gm:.15f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args.trace_table, args.figure14)
    if args.check_only:
        return
    write_csv(args.output, rows, OUTPUT_FIELDS)
    write_summary(args.summary, rows)


if __name__ == "__main__":
    main()
