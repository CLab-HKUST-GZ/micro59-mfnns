#!/usr/bin/env python3
"""Run the frozen 21-row Figure 14 CAGRA manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from build_indexes import index_name


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def gpu_processes(index: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(index),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip())
    return result.stdout.strip()


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "params" / "cagra.csv")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=str(root / "CAGRA" / "python"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = Path(__file__).with_name("benchmark.py")
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    selected = {item for item in args.only.split(",") if item}
    if selected:
        rows = [row for row in rows if row["id"] in selected or row["dataset"] in selected]
    if not rows:
        raise SystemExit("no CAGRA rows selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    command_log = args.output_root / "commands.log"
    status_log = args.output_root / "status.jsonl"
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for ordinal, row in enumerate(rows, start=1):
        before = "" if args.dry_run else gpu_processes(args.gpu)
        if before and not args.allow_busy_gpu:
            raise RuntimeError(f"physical GPU {args.gpu} has compute processes: {before}")
        dataset = args.data_root / row["dataset"]
        ground_truth = dataset / ("gt_top100.ibin" if int(row["top_k"]) == 100 else "gt_top32.ibin")
        output = args.output_root / f"{row['id']}.json"
        command = [
            args.python, str(benchmark),
            "--index", str(args.index_root / index_name(row)),
            "--query", str(dataset / "query.fbin"),
            "--ground-truth", str(ground_truth),
            "--output", str(output),
            "--top-k", row["top_k"],
            "--itopk-size", row["itopk_size"],
            "--search-width", row["search_width"],
            "--warmup", row["n_warmup"],
            "--runs", row["n_runs"],
            "--target-batch-size", "1000",
        ]
        if truth(row["repeat_queries"]):
            command.append("--repeat-queries")
        if truth(row["mmap"]):
            command.append("--mmap")
        line = shlex.join(command)
        with command_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(f"[{ordinal}/{len(rows)}] {row['id']}\n{line}", flush=True)
        if args.dry_run:
            continue
        log = args.output_root / f"{row['id']}.log"
        result = subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write_text(result.stdout, encoding="utf-8")
        record = {
            "id": row["id"],
            "returncode": result.returncode,
            "status": "ok" if result.returncode == 0 else "failed",
            "foreign_processes_before": before,
            "foreign_processes_after": gpu_processes(args.gpu),
            "output": str(output),
            "target_qps": float(row["target_qps"]),
            "target_recall": float(row["target_recall"]),
        }
        with status_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if result.returncode:
            print(result.stdout, file=sys.stderr)
            return result.returncode
    print(f"CAGRA_CASES={len(rows)} STATUS={'DRY_RUN' if args.dry_run else 'PASS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
