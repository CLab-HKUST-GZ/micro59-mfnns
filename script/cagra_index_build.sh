#!/usr/bin/env bash
# Build and validate a reusable cuVS CAGRA index from a binary float32 dataset.
set -euo pipefail

task_usage() {
    cat <<'EOF'
Usage:
  script/cagra_index_build.sh \
    --base BASE.bin \
    --dataset-name NAME \
    --cache-dir DIR \
    --graph-degree N \
    [--intermediate-graph-degree N] \
    [--metric sqeuclidean] \
    [--build-algo nn_descent] \
    [--nn-descent-niter N] \
    [--refinement-rate FLOAT] \
    [--output FILE.cagra] \
    [--metadata FILE.json] \
    [--gpu INDEX] \
    [--python PYTHON] \
    [--force] \
    [--allow-busy-gpu] \
    [--checksum]

Input format:
  Little-endian FBIN: int32 n, int32 dim, followed by exactly
  n * dim float32 values. Supply the same prepared/normalized base file that
  will be used by the CAGRA search workflow.

Default cache name:
  NAME_gdN_igdN_metric-METRIC_algo-ALGO[...].cagra

Environment:
  CAGRA_PYTHON  Python executable containing numpy, cupy, and cuvs
                (default: python3)
  GPU_INDEX     Physical GPU index (default: 0; overridden by --gpu)

Example (the unified Wiki k=100 graph parameters):
  CAGRA_PYTHON=/path/to/cuvs-env/bin/python \
  script/cagra_index_build.sh \
    --base /path/to/wiki_base.normalized.fbin \
    --dataset-name wiki \
    --cache-dir /path/to/cagra_cache \
    --graph-degree 6 \
    --intermediate-graph-degree 12 \
    --metric sqeuclidean \
    --build-algo nn_descent \
    --gpu 1

Behavior:
  * Existing indexes are loaded and validated, then reported as cache hits.
  * New indexes are saved to a temporary path, reloaded, and atomically moved.
  * --force rebuilds and atomically replaces an existing index.
  * By default, the selected physical GPU must have no compute process.
EOF
}

task_python="${CAGRA_PYTHON:-python3}"
task_gpu="${GPU_INDEX:-0}"
task_allow_busy=0
task_args=("$@")

for ((task_i = 0; task_i < ${#task_args[@]}; task_i++)); do
    case "${task_args[$task_i]}" in
        -h|--help)
            task_usage
            exit 0
            ;;
        --python)
            ((task_i += 1))
            [[ $task_i -lt ${#task_args[@]} ]] || {
                echo "ERROR: --python requires a value" >&2
                exit 2
            }
            task_python="${task_args[$task_i]}"
            ;;
        --python=*)
            task_python="${task_args[$task_i]#*=}"
            ;;
        --gpu)
            ((task_i += 1))
            [[ $task_i -lt ${#task_args[@]} ]] || {
                echo "ERROR: --gpu requires a value" >&2
                exit 2
            }
            task_gpu="${task_args[$task_i]}"
            ;;
        --gpu=*)
            task_gpu="${task_args[$task_i]#*=}"
            ;;
        --allow-busy-gpu)
            task_allow_busy=1
            ;;
    esac
done

if [[ "$task_python" == */* ]]; then
    [[ -x "$task_python" ]] || {
        echo "ERROR: Python is not executable: $task_python" >&2
        exit 2
    }
else
    task_python="$(command -v "$task_python")" || {
        echo "ERROR: Python command not found: $task_python" >&2
        exit 2
    }
fi
[[ "$task_gpu" =~ ^[0-9]+$ ]] || {
    echo "ERROR: --gpu must be a non-negative integer: $task_gpu" >&2
    exit 2
}
command -v nvidia-smi >/dev/null || {
    echo "ERROR: nvidia-smi is required" >&2
    exit 2
}
nvidia-smi -i "$task_gpu" --query-gpu=index,name,uuid,memory.used,memory.total \
    --format=csv >/dev/null

task_foreign="$(
    nvidia-smi -i "$task_gpu" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "$task_foreign" && "$task_allow_busy" -ne 1 ]]; then
    echo "ERROR: physical GPU $task_gpu already has compute processes:" >&2
    echo "$task_foreign" >&2
    echo "Use another GPU or explicitly pass --allow-busy-gpu." >&2
    exit 3
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$task_gpu"

"$task_python" - "${task_args[@]}" <<'PY'
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import cupy as cp
import numpy as np
import cuvs
from cuvs.neighbors import cagra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph-degree", type=int, required=True)
    parser.add_argument("--intermediate-graph-degree", type=int)
    parser.add_argument("--metric", default="sqeuclidean")
    parser.add_argument("--build-algo", default="nn_descent")
    parser.add_argument("--nn-descent-niter", type=int)
    parser.add_argument("--refinement-rate", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args()

    if "/" in args.dataset_name or args.dataset_name in {"", ".", ".."}:
        parser.error("--dataset-name must be a cache-safe name without '/'")
    if args.graph_degree <= 0:
        parser.error("--graph-degree must be positive")
    if args.intermediate_graph_degree is None:
        args.intermediate_graph_degree = args.graph_degree * 2
    if args.intermediate_graph_degree < args.graph_degree:
        parser.error("--intermediate-graph-degree must be >= --graph-degree")
    if args.nn_descent_niter is not None and args.nn_descent_niter <= 0:
        parser.error("--nn-descent-niter must be positive")
    if args.refinement_rate is not None and args.refinement_rate <= 0:
        parser.error("--refinement-rate must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_base(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Base file does not exist: {path}")
    header = np.fromfile(path, dtype=np.int32, count=2)
    if header.size != 2:
        raise ValueError(f"Invalid base header: {path}")
    n_rows, dim = (int(header[0]), int(header[1]))
    if n_rows <= 0 or dim <= 0:
        raise ValueError(f"Invalid base shape {n_rows} x {dim}: {path}")
    expected_bytes = 8 + n_rows * dim * np.dtype(np.float32).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Base size mismatch: expected {expected_bytes}, got {actual_bytes}: {path}"
        )
    vectors = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        offset=8,
        shape=(n_rows, dim),
    )
    return vectors, n_rows, dim, actual_bytes


def make_index_params(args: argparse.Namespace):
    common = {
        "metric": args.metric,
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
    }
    optional = {}
    if args.nn_descent_niter is not None:
        optional["nn_descent_niter"] = args.nn_descent_niter
    if args.refinement_rate is not None:
        optional["refinement_rate"] = args.refinement_rate

    failures = []
    for algo_key in ("build_algo", "graph_build_algo"):
        kwargs = {**common, algo_key: args.build_algo, **optional}
        try:
            return cagra.IndexParams(**kwargs), algo_key
        except TypeError as error:
            failures.append(f"{algo_key}: {error}")
    raise RuntimeError(
        "Installed cuVS cannot express the requested build algorithm/options; "
        "refusing to silently build a differently configured index. "
        + " | ".join(failures)
    )


def cache_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output.expanduser().resolve()
    name = (
        f"{args.dataset_name}_gd{args.graph_degree}_"
        f"igd{args.intermediate_graph_degree}_"
        f"metric-{args.metric}_algo-{args.build_algo}"
    )
    if args.nn_descent_niter is not None:
        name += f"_nniter-{args.nn_descent_niter}"
    if args.refinement_rate is not None:
        name += f"_refine-{args.refinement_rate:g}"
    return (args.cache_dir / f"{name}.cagra").expanduser().resolve()


def atomic_json(path: Path, payload) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


args = parse_args()
args.base = args.base.expanduser().resolve()
args.cache_dir = args.cache_dir.expanduser().resolve()
args.cache_dir.mkdir(parents=True, exist_ok=True)
output = cache_path(args)
if output.suffix != ".cagra":
    raise ValueError(f"Output must end in .cagra: {output}")
output.parent.mkdir(parents=True, exist_ok=True)
metadata_path = (
    args.metadata.expanduser().resolve()
    if args.metadata is not None
    else output.with_suffix(output.suffix + ".metadata.json")
)
metadata_path.parent.mkdir(parents=True, exist_ok=True)

base_vectors, n_rows, dim, base_bytes = inspect_base(args.base)
lock_path = output.with_suffix(output.suffix + ".lock")
lock_handle = lock_path.open("a+")
try:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError as error:
    raise RuntimeError(f"Another builder holds the lock: {lock_path}") from error

started = time.time()
cache_hit = output.exists() and not args.force
build_time_s = None
load_validation_time_s = None
params_key = None
temp_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")

try:
    if cache_hit:
        print(f"Loading existing CAGRA index: {output}", flush=True)
        load_started = time.time()
        index = cagra.load(str(output))
        cp.cuda.Device().synchronize()
        load_validation_time_s = time.time() - load_started
    else:
        params, params_key = make_index_params(args)
        print(
            f"Building CAGRA index: base={args.base} shape={n_rows}x{dim} "
            f"gd={args.graph_degree} igd={args.intermediate_graph_degree} "
            f"metric={args.metric} build_algo={args.build_algo}",
            flush=True,
        )
        base_gpu = cp.asarray(base_vectors)
        cp.cuda.Device().synchronize()
        build_started = time.time()
        index = cagra.build(params, base_gpu)
        cp.cuda.Device().synchronize()
        build_time_s = time.time() - build_started

        if temp_output.exists():
            temp_output.unlink()
        print(f"Saving temporary index: {temp_output}", flush=True)
        cagra.save(str(temp_output), index, include_dataset=True)

        print("Reloading temporary index for validation", flush=True)
        load_started = time.time()
        validated_index = cagra.load(str(temp_output))
        cp.cuda.Device().synchronize()
        load_validation_time_s = time.time() - load_started
        del validated_index
        os.replace(temp_output, output)
        print(f"Committed index atomically: {output}", flush=True)

    device = cp.cuda.runtime.getDeviceProperties(0)
    device_name = device["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    payload = {
        "status": "cache_hit" if cache_hit else "built",
        "cache_hit": cache_hit,
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "physical_gpu_index": args.gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": device_name,
        "base_path": str(args.base),
        "base_rows": n_rows,
        "base_dim": dim,
        "base_bytes": base_bytes,
        "base_sha256": sha256_file(args.base) if args.checksum else None,
        "index_path": str(output),
        "index_bytes": output.stat().st_size,
        "index_sha256": sha256_file(output) if args.checksum else None,
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
        "metric": args.metric,
        "build_algo": args.build_algo,
        "build_algo_parameter_key": params_key,
        "nn_descent_niter": args.nn_descent_niter,
        "refinement_rate": args.refinement_rate,
        "include_dataset": True,
        "force": args.force,
        "build_time_s": build_time_s,
        "load_validation_time_s": load_validation_time_s,
        "elapsed_s": time.time() - started,
        "python": sys.version,
        "cupy_version": cp.__version__,
        "cuvs_version": cuvs.__version__,
    }
    atomic_json(metadata_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
finally:
    if temp_output.exists():
        temp_output.unlink()
PY
