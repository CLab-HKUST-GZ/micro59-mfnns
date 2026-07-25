#!/usr/bin/env python3
"""Summarize Figure 18 reruns recorded by simulator/memory/run_yaml_case.py."""

from __future__ import annotations

import argparse
import csv
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PROVENANCE = SCRIPT_DIR / "data/simulator_provenance.tsv"
FREQ_HZ = 2_400_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recall-digits", type=int, default=2)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_record(path: Path) -> dict[str, str]:
    record: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        record[key] = value
    return record


def first_stat(text: str, key: str) -> float:
    match = re.search(
        rf"^\s*{re.escape(key)}:\s*([-+0-9.eE]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Missing {key}")
    return float(match.group(1))


def round_recall(value: float, digits: int) -> str:
    quantum = Decimal(1).scaleb(-digits)
    return str(
        Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    )


def main() -> None:
    args = parse_args()
    provenance = read_tsv(args.provenance)
    by_config = {
        str((REPO_ROOT / row["portable_config_ref"]).resolve()): row
        for row in provenance
    }
    output = args.output or args.result_root / "rerun_summary.tsv"
    rows: list[dict[str, str]] = []
    for record_path in sorted(args.result_root.rglob("run_record.tsv")):
        record = read_record(record_path)
        config = by_config.get(str(Path(record["yaml_path"]).resolve()))
        if config is None:
            continue
        stats_path = Path(record["stat_path_resolved"])
        result = {
            "point_id": config["point_id"],
            "dataset": config["dataset"],
            "recall_tag": config["recall_tag"],
            "method": config["method"],
            "config_status": config["config_status"],
            "runner_status": record["status"],
            "stats_status": "missing",
            "recall_raw": "",
            "recall_rounded": "",
            "cycle": "",
            "qps_2p4ghz": "",
            "plot_recall_raw": config["plot_recall_raw"],
            "plot_cycle": config["plot_cycle"],
            "recall_delta_rerun_minus_plot": "",
            "cycle_delta_rerun_minus_plot": "",
            "yaml_path": record["yaml_path"],
            "stats_path": str(stats_path),
            "run_record": str(record_path),
        }
        if stats_path.is_file():
            text = stats_path.read_text(encoding="utf-8", errors="ignore")
            recall = first_stat(text, "s_recall_rate")
            cycle = int(first_stat(text, "s_mem_cycle"))
            nq = int(first_stat(text, "s_num_query"))
            qps = nq * FREQ_HZ / cycle
            result.update(
                {
                    "stats_status": "complete",
                    "recall_raw": f"{recall:.10g}",
                    "recall_rounded": round_recall(recall, args.recall_digits),
                    "cycle": str(cycle),
                    "qps_2p4ghz": f"{qps:.6f}",
                    "recall_delta_rerun_minus_plot": f"{recall-float(config['plot_recall_raw']):.10g}",
                    "cycle_delta_rerun_minus_plot": str(cycle-int(config["plot_cycle"])),
                }
            )
        rows.append(result)
    if not rows:
        raise ValueError(f"No Figure 18 run records found below {args.result_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rerun rows to {output}")


if __name__ == "__main__":
    main()
