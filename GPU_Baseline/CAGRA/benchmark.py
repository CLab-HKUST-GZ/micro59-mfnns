#!/usr/bin/env python3
"""Run one CAGRA search point against a serialized index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import time

import cupy as cp
import numpy as np
from cuvs.neighbors import cagra


def read_vectors(path: Path, mmap: bool):
    with path.open("rb") as handle:
        rows, dim = struct.unpack("<II", handle.read(8))
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"vector size mismatch: {path}")
    if mmap:
        return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))
    return np.fromfile(path, dtype="<f4", offset=8).reshape(rows, dim)


def read_labels(path: Path):
    with path.open("rb") as handle:
        rows, width = struct.unpack("<II", handle.read(8))
    expected = 8 + rows * width * 4
    if path.stat().st_size != expected:
        raise ValueError(f"label size mismatch: {path}")
    return np.fromfile(path, dtype="<u4", offset=8).reshape(rows, width)


def search_params(args):
    values = {
        "itopk_size": args.itopk_size,
        "search_width": args.search_width,
        "max_iterations": 0,
        "algo": "auto",
    }
    try:
        return cagra.SearchParams(**values)
    except TypeError as error:
        raise RuntimeError(
            "installed cuVS cannot express the frozen Figure 14 search "
            f"parameters {values}: {error}"
        ) from error


def recall_at_k(predicted: np.ndarray, expected: np.ndarray, k: int) -> float:
    total = 0.0
    for predicted_row, expected_row in zip(predicted[:, :k], expected[:, :k]):
        total += len(set(map(int, predicted_row)) & set(map(int, expected_row))) / k
    return total / predicted.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--itopk-size", type=int, required=True)
    parser.add_argument("--search-width", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--target-batch-size", type=int, default=1000)
    parser.add_argument("--repeat-queries", action="store_true")
    parser.add_argument("--mmap", action="store_true")
    args = parser.parse_args()
    for name in ("top_k", "itopk_size", "search_width", "runs", "target_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    for path in (args.index, args.query, args.ground_truth):
        if not path.is_file():
            raise FileNotFoundError(path)
    queries = read_vectors(args.query, args.mmap)
    labels = read_labels(args.ground_truth)
    if queries.shape[0] != labels.shape[0]:
        raise ValueError("query and ground-truth row counts differ")
    if labels.shape[1] < args.top_k:
        raise ValueError("ground truth is narrower than --top-k")
    original_queries = queries.shape[0]
    if original_queries < args.target_batch_size:
        if not args.repeat_queries:
            raise ValueError(
                "query count is below target batch size; pass --repeat-queries "
                "only for recorded repeated-query cases"
            )
        copies = (args.target_batch_size + original_queries - 1) // original_queries
        queries = np.tile(queries, (copies, 1))[: args.target_batch_size]
        labels = np.tile(labels, (copies, 1))[: args.target_batch_size]
    elif original_queries > args.target_batch_size:
        queries = queries[: args.target_batch_size]
        labels = labels[: args.target_batch_size]

    load_started = time.perf_counter()
    index = cagra.load(str(args.index))
    cp.cuda.Device().synchronize()
    load_seconds = time.perf_counter() - load_started
    query_device = cp.asarray(queries)
    params = search_params(args)
    for _ in range(args.warmup):
        distances, indices = cagra.search(params, index, query_device, args.top_k)
        cp.cuda.Device().synchronize()
    started = time.perf_counter()
    for _ in range(args.runs):
        distances, indices = cagra.search(params, index, query_device, args.top_k)
        cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    latency_seconds = elapsed / args.runs
    predicted = cp.asnumpy(indices)
    recall = recall_at_k(predicted, labels, args.top_k)
    payload = {
        "index": str(args.index.resolve()),
        "query": str(args.query.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "query_rows_original": original_queries,
        "query_rows_measured": int(queries.shape[0]),
        "top_k": args.top_k,
        "itopk_size": args.itopk_size,
        "search_width": args.search_width,
        "warmup": args.warmup,
        "runs": args.runs,
        "repeat_queries": args.repeat_queries,
        "mmap": args.mmap,
        "index_load_seconds": load_seconds,
        "latency_ms": latency_seconds * 1000.0,
        "qps": queries.shape[0] / latency_seconds,
        "recall": recall,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
