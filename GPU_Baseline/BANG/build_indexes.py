#!/usr/bin/env python3
"""Build every unique BANG graph referenced by the frozen parameter CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shlex
import subprocess


def index_stem(row: dict[str, str]) -> str:
    return (
        f"{row['dataset']}_R{row['graph_R']}_Lb{row['build_L']}_"
        f"PQ{row['pq_chunks']}"
    )


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "params" / "bang.csv")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--builder", required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--build-memory-gb", type=int, default=64)
    parser.add_argument("--only", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = Path(__file__).resolve().parents[2] / "script" / "bang_index_build.sh"
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    selected = {item for item in args.only.split(",") if item}
    if selected:
        rows = [row for row in rows if row["id"] in selected or row["dataset"] in selected]
    unique = {}
    for row in rows:
        unique[index_stem(row)] = row
    if not unique:
        raise SystemExit("no BANG rows selected")
    for stem, row in unique.items():
        command = [
            str(script),
            "--base", str(args.data_root / row["dataset"] / "base.fbin"),
            "--dataset-name", row["dataset"],
            "--output-dir", str(args.index_root / row["dataset"] / "BANG"),
            "--graph-degree", row["graph_R"],
            "--build-l", row["build_L"],
            "--pq-chunks", row["pq_chunks"],
            "--bf-entries", row["bf_entries"],
            "--build-memory-gb", str(args.build_memory_gb),
            "--threads", str(args.threads),
            "--build-disk-index", args.builder,
            "--python", args.python,
        ]
        if args.force:
            command.append("--force")
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    print(f"BANG_INDEX_CONFIGS={len(unique)} STATUS={'DRY_RUN' if args.dry_run else 'PASS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
