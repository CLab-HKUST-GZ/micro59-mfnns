#!/usr/bin/env python3
"""Run the frozen 21-row Figure 14 BANG manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import struct
import subprocess
import sys

from build_indexes import index_stem


RESULT_ROW = re.compile(
    r"^\s*(?P<L>\d+)\s+(?P<time_ms>[0-9.]+)\s+"
    r"(?P<qps>[0-9.]+)\s+(?P<recall_percent>[0-9.]+)\s*$"
)


def contract_id(row: dict[str, str]) -> str:
    return f"R{row['graph_R']}_BF{row['bf_entries']}"


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


def validate_inputs(query: Path, ground_truth: Path, count: int, top_k: int) -> None:
    with query.open("rb") as handle:
        query_rows, query_dim = struct.unpack("<II", handle.read(8))
    if query_rows < count or query_dim <= 0:
        raise ValueError(f"invalid query shape {(query_rows, query_dim)}: {query}")
    if query.stat().st_size != 8 + query_rows * query_dim * 4:
        raise ValueError(f"query size mismatch: {query}")
    with ground_truth.open("rb") as handle:
        gt_rows, gt_width = struct.unpack("<II", handle.read(8))
    if gt_rows < count or gt_width < top_k:
        raise ValueError(f"invalid BANG GT shape {(gt_rows, gt_width)}: {ground_truth}")
    expected = 8 + gt_rows * gt_width * 8
    if ground_truth.stat().st_size != expected:
        raise ValueError(f"BANG GT size mismatch: {ground_truth}")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "params" / "bang.csv")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda-12.8"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--omp-threads", type=int, default=64)
    parser.add_argument("--only", default="")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="")))
    selected = {item for item in args.only.split(",") if item}
    if selected:
        rows = [row for row in rows if row["id"] in selected or row["dataset"] in selected]
    if not rows:
        raise SystemExit("no BANG rows selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    command_log = args.output_root / "commands.log"
    status_log = args.output_root / "status.jsonl"
    for ordinal, row in enumerate(rows, start=1):
        contract = args.binary_root / contract_id(row)
        binary = contract / "bang_search"
        prefix = (
            args.index_root / row["dataset"] / "BANG" / f"{index_stem(row)}_bang"
        )
        dataset = args.data_root / row["dataset"]
        query = dataset / "query.fbin"
        ground_truth = dataset / "gt_top100.bang.bin"
        output = args.output_root / f"{row['id']}.json"
        log = args.output_root / f"{row['id']}.log"
        display = [
            str(binary), str(prefix), str(query), str(ground_truth),
            row["query_count"], row["top_k"], "float", "l2",
        ]
        line = (
            f"printf {shlex.quote(row['search_L'] + chr(10) + 'n' + chr(10))} | "
            + shlex.join(display)
        )
        with command_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(f"[{ordinal}/{len(rows)}] {row['id']}\n{line}", flush=True)
        if args.dry_run:
            continue
        for path in (binary, contract / "libbang.so", query, ground_truth):
            if not path.is_file():
                raise FileNotFoundError(path)
        for suffix in (
            "_disk.bin", "_disk_metadata.bin", "_pq_compressed.bin", "_pq_pivots.bin"
        ):
            path = Path(str(prefix) + suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
        validate_inputs(query, ground_truth, int(row["query_count"]), int(row["top_k"]))
        before = gpu_processes(args.gpu)
        if before and not args.allow_busy_gpu:
            raise RuntimeError(f"physical GPU {args.gpu} has compute processes: {before}")
        env = os.environ.copy()
        env["PATH"] = f"{args.cuda_root / 'bin'}:{env.get('PATH', '')}"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["OMP_NUM_THREADS"] = str(args.omp_threads)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_PROC_BIND"] = "false"
        env["LD_LIBRARY_PATH"] = (
            f"{contract}:{args.cuda_root / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
        )
        result = subprocess.run(
            display,
            input=f"{row['search_L']}\nn\n",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write_text(result.stdout, encoding="utf-8")
        measurements = []
        for text in result.stdout.splitlines():
            match = RESULT_ROW.match(text)
            if match and int(match.group("L")) == int(row["search_L"]):
                measurements.append(
                    {
                        "time_ms": float(match.group("time_ms")),
                        "qps": float(match.group("qps")),
                        "recall": float(match.group("recall_percent")) / 100.0,
                    }
                )
        if result.returncode != 0:
            failure = {
                "id": row["id"],
                "returncode": result.returncode,
                "status": "failed",
                "log": str(log),
                "foreign_processes_before": before,
                "foreign_processes_after": gpu_processes(args.gpu),
            }
            with status_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")
            print(json.dumps(failure, sort_keys=True), file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            return result.returncode
        if not measurements:
            raise RuntimeError(f"BANG output contained no result rows; see {log}")
        warm = measurements[1:] if len(measurements) > 1 else measurements
        payload = {
            "id": row["id"],
            "dataset": row["dataset"],
            "contract": contract_id(row),
            "graph_R": int(row["graph_R"]),
            "build_L": int(row["build_L"]),
            "pq_chunks": int(row["pq_chunks"]),
            "bf_entries": int(row["bf_entries"]),
            "search_L": int(row["search_L"]),
            "top_k": int(row["top_k"]),
            "measurements": measurements,
            "warm_qps_median": statistics.median(item["qps"] for item in warm),
            "recall": measurements[-1]["recall"],
            "target_qps": float(row["target_qps"]),
            "target_recall": float(row["target_recall"]),
            "returncode": result.returncode,
            "foreign_processes_before": before,
            "foreign_processes_after": gpu_processes(args.gpu),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        with status_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)
    print(f"BANG_CASES={len(rows)} STATUS={'DRY_RUN' if args.dry_run else 'PASS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
