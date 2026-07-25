#!/usr/bin/env python3
"""Build the portable Recall@10 energy-efficiency data for Figure 16."""

from __future__ import annotations

import argparse
import csv
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
    load_figure14_lookup,
    read_csv,
    write_csv,
)


DEFAULT_EXTERNAL_POWER = SCRIPT_DIR / "data/external_power.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "data/figure16_energy_efficiency.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "output/figure16_summary.tsv"

METHOD_ORDER = [
    "CPU",
    "CAGRA",
    "BANG",
    "ANSMET",
    "NMP-Base",
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
    "qps_source_design",
    "trace_design",
    "qps",
    "recall",
    "power_w",
    "total_energy_nj",
    "energy_efficiency_qps_per_w",
    "energy_efficiency_norm_to_cpu",
    "config_ref",
    "trace_table_ref",
    "data_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-table", type=Path, default=DEFAULT_TRACE_TABLE)
    parser.add_argument("--figure14", type=Path, default=DEFAULT_FIGURE14_CSV)
    parser.add_argument(
        "--external-power", type=Path, default=DEFAULT_EXTERNAL_POWER
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def load_external_power(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    required = {"dataset_key", "dataset_short", "method", "power_w", "source_ref"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"External-power CSV is missing columns: {sorted(missing)}")
    expected = {
        (dataset_key, method)
        for dataset_key, _, _ in DATASET_ORDER
        for method in ("CAGRA", "BANG")
    }
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset_key"], row["method"])
        if key in lookup:
            raise ValueError(f"Duplicate external-power row: {key}")
        if Path(row["source_ref"]).is_absolute():
            raise ValueError(f"Absolute external-power source: {row['source_ref']}")
        if float(row["power_w"]) <= 0:
            raise ValueError(f"Invalid external power for {key}")
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError("External-power matrix does not contain 7x2 rows")
    return lookup


def build_rows(
    trace_path: Path, figure14_path: Path, external_power_path: Path
) -> list[dict[str, object]]:
    simulator_rows = build_energy_rows(trace_path, figure14_path)
    simulator = {
        (str(row["dataset_key"]), str(row["method"])): row
        for row in simulator_rows
    }
    figure14 = load_figure14_lookup(figure14_path)
    external = load_external_power(external_power_path)
    rows: list[dict[str, object]] = []

    for dataset_key, dataset_label, dataset_short in DATASET_ORDER:
        cpu_efficiency = float(
            simulator[(dataset_key, "CPU")]["energy_efficiency_qps_per_w"]
        )
        for method in METHOD_ORDER:
            if method in ("CAGRA", "BANG"):
                design = "gpu_cagra" if method == "CAGRA" else "bang"
                qps_row = figure14[(dataset_key, design)]
                power_row = external[(dataset_key, method)]
                qps = float(qps_row["qps"])
                power_w = float(power_row["power_w"])
                efficiency = qps / power_w
                row: dict[str, object] = {
                    "top_k": "k10",
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                    "dataset_short": dataset_short,
                    "method": method,
                    "qps_source_design": design,
                    "trace_design": "",
                    "qps": qps,
                    "recall": float(qps_row["recall"]),
                    "power_w": power_w,
                    "total_energy_nj": "",
                    "energy_efficiency_qps_per_w": efficiency,
                    "config_ref": "",
                    "trace_table_ref": power_row["source_ref"],
                    "data_status": qps_row["data_status"],
                }
            else:
                source = simulator[(dataset_key, method)]
                row = {field: source.get(field, "") for field in OUTPUT_FIELDS}
            row["energy_efficiency_norm_to_cpu"] = (
                float(row["energy_efficiency_qps_per_w"]) / cpu_efficiency
            )
            rows.append(row)

    expected = len(DATASET_ORDER) * len(METHOD_ORDER)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} Figure 16 rows, found {len(rows)}")
    by_key = {
        (str(row["dataset_key"]), str(row["method"])): row for row in rows
    }
    for dataset_key, _, _ in DATASET_ORDER:
        base = by_key[(dataset_key, "NMP-Base-ET")]
        mfnns = by_key[(dataset_key, "MFNNS")]
        if base["qps"] != mfnns["qps"]:
            raise ValueError(f"{dataset_key}: Base-ET/MFNNS QPS mismatch")
        if float(base["energy_efficiency_qps_per_w"]) >= float(
            mfnns["energy_efficiency_qps_per_w"]
        ):
            raise ValueError(
                f"{dataset_key}: Base-ET must have lower QPS/W than MFNNS"
            )
    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    by_method = {
        method: [row for row in rows if row["method"] == method]
        for method in METHOD_ORDER
    }
    lines = ["metric\tvalue"]
    for method in METHOD_ORDER:
        gm = geometric_mean(
            [float(row["energy_efficiency_norm_to_cpu"]) for row in by_method[method]]
        )
        lines.append(f"geomean_norm_to_cpu:{method}\t{gm:.15f}")
    lookup = {
        (str(row["dataset_key"]), str(row["method"])): float(
            row["energy_efficiency_qps_per_w"]
        )
        for row in rows
    }
    for baseline in (
        "CPU",
        "CAGRA",
        "BANG",
        "ANSMET",
        "NMP-Base",
        "NMP-FPMA",
        "NMP-FPSA",
        "NMP-Base-ET",
        "NMP-FPSA-ET",
    ):
        ratios = [
            lookup[(dataset_key, "MFNNS")] / lookup[(dataset_key, baseline)]
            for dataset_key, _, _ in DATASET_ORDER
        ]
        lines.append(f"geomean_mfnns_over:{baseline}\t{geometric_mean(ratios):.15f}")
        lines.append(f"min_mfnns_over:{baseline}\t{min(ratios):.15f}")
        lines.append(f"max_mfnns_over:{baseline}\t{max(ratios):.15f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args.trace_table, args.figure14, args.external_power)
    if args.check_only:
        return
    write_csv(args.output, rows, OUTPUT_FIELDS)
    write_summary(args.summary, rows)


if __name__ == "__main__":
    main()
