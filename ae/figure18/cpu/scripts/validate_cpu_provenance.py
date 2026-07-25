#!/usr/bin/env python3
"""Validate and export the Figure 18 CPU raw-trial provenance mapping."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from decimal import Decimal
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CPU_DIR = SCRIPT_DIR.parent
FIGURE_DIR = CPU_DIR.parent
DEFAULT_FIGURE_DATA = FIGURE_DIR / "data" / "figure18_recall_qps.csv"
DEFAULT_RAW = CPU_DIR / "data" / "fresh_raw_all_trials.tsv"
DEFAULT_OUTPUT = CPU_DIR / "data" / "figure18_cpu_selected_trials.tsv"

EXPECTED_PANEL_COUNTS = {
    ("t2i1b", "r10"): 8,
    ("deep1b", "r10"): 8,
    ("t2i1b", "r100"): 8,
    ("deep1b", "r100"): 7,
}

FIELDS = [
    "dataset",
    "recall_tag",
    "k",
    "ef",
    "trial",
    "recall",
    "qps",
    "threads",
    "nq",
    "query_ops",
    "warmup_queries",
    "avg_results_per_query",
    "raw_source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-data", type=Path, default=DEFAULT_FIGURE_DATA)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the existing mapping instead of rewriting it.",
    )
    return parser.parse_args()


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def build_mapping(
    figure_rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    if len(raw_rows) != 96:
        raise ValueError(f"Expected 96 CPU raw trials, found {len(raw_rows)}")

    groups = Counter(
        (row["dataset"], row["metric"], row["ef"])
        for row in raw_rows
    )
    if len(groups) != 32 or set(groups.values()) != {3}:
        raise ValueError(
            f"Expected 32 CPU points with 3 trials each, got {dict(groups)}"
        )

    cpu_rows = [row for row in figure_rows if row["method"] == "cpu"]
    if len(cpu_rows) != 31:
        raise ValueError(f"Expected 31 plotted CPU rows, found {len(cpu_rows)}")

    mapping: list[dict[str, str]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for figure_row in cpu_rows:
        dataset = (
            "deep1b"
            if figure_row["dataset"] == "deep1b"
            else "t2i1b_norml2"
        )
        metric = "R@10" if figure_row["recall_tag"] == "r10" else "R@100"
        matches = [
            raw
            for raw in raw_rows
            if raw["dataset"] == dataset
            and raw["metric"] == metric
            and Decimal(raw["recall"]) == Decimal(figure_row["recall_raw"])
            and Decimal(raw["qps"]) == Decimal(figure_row["qps"])
        ]
        if len(matches) != 1:
            raise ValueError(
                "CPU plot row does not map uniquely to a raw trial: "
                f"{figure_row}; matches={matches}"
            )
        raw = matches[0]
        counts[(figure_row["dataset"], figure_row["recall_tag"])] += 1
        mapping.append(
            {
                "dataset": figure_row["dataset"],
                "recall_tag": figure_row["recall_tag"],
                "k": raw["k"],
                "ef": raw["ef"],
                "trial": raw["trial"],
                "recall": raw["recall"],
                "qps": raw["qps"],
                "threads": raw["threads"],
                "nq": raw["nq"],
                "query_ops": raw["query_ops"],
                "warmup_queries": raw["warmup_queries"],
                "avg_results_per_query": raw["avg_results_per_query"],
                "raw_source_file": Path(raw["source_file"]).name,
            }
        )

    if dict(counts) != EXPECTED_PANEL_COUNTS:
        raise ValueError(
            f"Unexpected CPU panel counts: expected={EXPECTED_PANEL_COUNTS}, "
            f"got={dict(counts)}"
        )
    return mapping


def write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    figure_rows = read_rows(args.figure_data, ",")
    raw_rows = read_rows(args.raw, "\t")
    expected = build_mapping(figure_rows, raw_rows)
    if args.check_only:
        actual = read_rows(args.output, "\t")
        if actual != expected:
            raise ValueError("Stored CPU selected-trial mapping is stale")
    else:
        write_mapping(args.output, expected)
    trial_counts = Counter(row["trial"] for row in expected)
    print(
        "Validated Figure 18 CPU provenance: "
        f"31 plotted rows -> 96 raw trials; selected trials={dict(trial_counts)}"
    )


if __name__ == "__main__":
    main()
