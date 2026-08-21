#!/usr/bin/env python3
"""Build every unique CAGRA graph referenced by the frozen parameter CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shlex
import subprocess

def index_name(row: dict[str, str]) -> str:
    algorithm = row["graph_build_algo"] or "default"
    return (
        f"{row['dataset_arg']}_gd{row['graph_degree']}_"
        f"igd{row['intermediate_graph_degree']}_metric-{row['metric']}_"
        f"algo-{algorithm}.cagra"
    )


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "params" / "cagra.csv")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--python", default=str(root / "CAGRA" / "python"))
    parser.add_argument("--builder", type=Path, default=root.parent / "script" / "cagra_index_build.sh")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    selected = {item for item in args.only.split(",") if item}
    if selected:
        rows = [row for row in rows if row["id"] in selected or row["dataset"] in selected]
    unique = {}
    for row in rows:
        unique[index_name(row)] = row
    if not unique:
        raise SystemExit("no CAGRA rows selected")
    for name, row in unique.items():
        base = args.data_root / row["dataset"] / "base.fbin"
        output = args.index_root / name
        command = [
            str(args.builder),
            "--base", str(base),
            "--dataset-name", row["dataset_arg"],
            "--cache-dir", str(args.index_root),
            "--output", str(output),
            "--graph-degree", row["graph_degree"],
            "--intermediate-graph-degree", row["intermediate_graph_degree"],
            "--metric", row["metric"],
            "--build-algo", row["graph_build_algo"] or "default",
            "--gpu", str(args.gpu),
            "--python", args.python,
        ]
        if args.allow_busy_gpu:
            command.append("--allow-busy-gpu")
        if args.force:
            command.append("--force")
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    print(f"CAGRA_INDEX_CONFIGS={len(unique)} STATUS={'DRY_RUN' if args.dry_run else 'PASS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
