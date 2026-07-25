#!/usr/bin/env python3
"""Summarize a completed Figure 21 rerun and compare it with frozen results."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FROZEN_SWEEP = SCRIPT_DIR / "data/figure21_sweep_results.tsv"
ANSMET_STATS = SCRIPT_DIR / "data/ansmet_stats"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

OUTPUT_FIELDS = [
    "method",
    "case_name",
    "ef_search",
    "queue_size",
    "recall",
    "s_mem_cycle",
    "frozen_recall",
    "frozen_s_mem_cycle",
    "recall_delta",
    "cycle_delta",
    "status",
    "stats_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def extract(text: str, key: str) -> str:
    match = re.search(
        rf"^\s{{2}}{re.escape(key)}:\s*({FLOAT_PATTERN})\s*$",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else ""


def frozen_rows() -> dict[tuple[str, int, int], tuple[float, float]]:
    rows: dict[tuple[str, int, int], tuple[float, float]] = {}
    for row in read_tsv(FROZEN_SWEEP):
        rows[("mfnns", int(row["ef_search"]), int(row["queue_size"]))] = (
            float(row["recall"]),
            float(row["s_mem_cycle"]),
        )
    for ef_search in (20, 30, 40):
        path = ANSMET_STATS / f"t2i1m_normalized_k10_ef{ef_search}_stats.yml"
        text = path.read_text(encoding="utf-8")
        rows[("ansmet", ef_search, 33)] = (
            float(extract(text, "s_recall_rate")),
            float(extract(text, "s_mem_cycle")),
        )
    return rows


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    run_summary = result_root / "runs/summary.tsv"
    if not run_summary.is_file():
        raise FileNotFoundError(f"Missing run summary: {run_summary}")

    frozen = frozen_rows()
    output_rows: list[dict[str, str]] = []
    for run_row in read_tsv(run_summary):
        yaml_path = Path(run_row["yaml_path"])
        yaml_text = yaml_path.read_text(encoding="utf-8")
        method = yaml_path.parent.name
        ef_search = int(extract(yaml_text, "ef_search"))
        queue_size = int(extract(yaml_text, "dualQueueLowerBoundQueueSize"))
        stats_path = Path(run_row["stat_path_resolved"])
        recall = ""
        cycle = ""
        status = "MISSING_STATS"
        if stats_path.is_file():
            stats_text = stats_path.read_text(encoding="utf-8")
            recall = extract(stats_text, "s_recall_rate")
            cycle = extract(stats_text, "s_mem_cycle")
            status = "PASS" if recall and cycle else "MISSING_METRIC"

        frozen_recall, frozen_cycle = frozen[(method, ef_search, queue_size)]
        output_rows.append(
            {
                "method": method,
                "case_name": yaml_path.stem,
                "ef_search": str(ef_search),
                "queue_size": str(queue_size),
                "recall": recall or "NA",
                "s_mem_cycle": cycle or "NA",
                "frozen_recall": f"{frozen_recall:.17g}",
                "frozen_s_mem_cycle": f"{frozen_cycle:.17g}",
                "recall_delta": (
                    f"{float(recall) - frozen_recall:.17g}" if recall else "NA"
                ),
                "cycle_delta": (
                    f"{float(cycle) - frozen_cycle:.17g}" if cycle else "NA"
                ),
                "status": status,
                "stats_path": str(stats_path),
            }
        )

    output_path = result_root / "rerun_summary.tsv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    passed = sum(row["status"] == "PASS" for row in output_rows)
    print(f"SUMMARY_OK rows={len(output_rows)} pass={passed} output={output_path}")


if __name__ == "__main__":
    main()
