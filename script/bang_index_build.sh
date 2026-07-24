#!/usr/bin/env bash
# Build a DiskANN/PipeANN PQ graph and convert it to BANG_Base index files.
set -euo pipefail

task_usage() {
    cat <<'EOF'
Usage:
  script/bang_index_build.sh \
    --base BASE.fbin \
    --dataset-name NAME \
    --output-dir DIR \
    [--preset deep1b-qd64] \
    [--builder-api pipeann|diskann] \
    [--graph-degree N] \
    [--build-l N] \
    [--pq-chunks N] \
    [--build-memory-gb N] \
    [--search-dram-budget-gb N] \
    [--indexing-ram-budget-gb N] \
    [--build-pq-bytes N] \
    [--quantized-dim N] \
    [--pq-disk-bytes N] \
    [--threads N] \
    [--expect-points N] \
    [--bf-entries N] \
    [--build-disk-index FILE] \
    [--python PYTHON] \
    [--allow-non-normalized|--require-normalized] \
    [--staging-dir DIR] \
    [--keep-staging-on-failure|--discard-staging-on-failure] \
    [--force] \
    [--dry-run]

Input:
  Little-endian FBIN: uint32 n, uint32 dim, then exactly n * dim float32
  values. The reproduction datasets are unit-normalized; this is checked by
  default because their inner-product ground truth was reused for L2 search.

Build pipeline:
  1. Build the DiskANN/PipeANN PQ graph. Two command-line APIs are supported:
     pipeann:
       build_disk_index float BASE PREFIX R L PQ MEMORY_GB THREADS l2 pq
     diskann:
       build_disk_index --data_type float --dist_fn l2 \
         --data_path BASE --index_path_prefix PREFIX \
         -R R -L L -B SEARCH_GB -M INDEX_GB -T THREADS \
         --PQ_disk_bytes PQ_DISK --build_PQ_bytes BUILD_PQ --QD QD
  2. Convert PREFIX_disk.index to BANG's disk.bin + disk_metadata.bin.
  3. Rewrap the PQ-pivots header and create the BANG prefix symlinks.

Defaults from the recorded seven-dataset R16 baseline:
  builder-api=pipeann, R=16, L=64, PQ=128,
  build memory=64 GiB, threads=64,
  BF_ENTRIES=399887. Deep10M used PQ=96.

The deep1b-qd64 preset reproduces the successful one-billion-point build:
  builder-api=diskann, R=64, L=100, search DRAM budget=4 GiB,
  indexing RAM budget=32 GiB, threads=24, build-PQ=64, QD=64,
  PQ-disk-bytes=0, expect-points=1000000000, normalization check disabled,
  and failed staging directories preserved for diagnosis/resume.

Output prefixes:
  pipeann:
  DIR/NAME_R<R>_Lb<L>_PQ<PQ>
  DIR/NAME_R<R>_Lb<L>_PQ<PQ>_bang
  diskann:
  DIR/NAME_R<R>_L<L>_QD<QD>
  DIR/NAME_R<R>_L<L>_QD<QD>_bang
  If build-PQ differs from QD, _BPQ<BUILD_PQ> is included before _QD<QD>.

Environment:
  BANG_BUILD_DISK_INDEX  DiskANN/PipeANN build_disk_index executable
                        (default: build_disk_index from PATH)
  BANG_PYTHON            Python with numpy (default: python3)
  BANG_BUILD_MEMORY_GB   PipeANN builder memory budget (default: 64)
  BANG_BUILD_THREADS     Builder threads (default: 64)

Baseline example:
  BANG_BUILD_DISK_INDEX=/path/to/PipeANN/build/tests/build_disk_index \
  script/bang_index_build.sh \
    --base /path/to/wiki_base.normalized.fbin \
    --dataset-name wiki1M \
    --output-dir /local-ssd/$USER/bang/index

Recorded Wiki k=100 tuned graph:
  BANG_BUILD_DISK_INDEX=/path/to/build_disk_index \
  script/bang_index_build.sh \
    --base /path/to/wiki_base.normalized.fbin \
    --dataset-name wiki1M \
    --output-dir /local-ssd/$USER/bang/index \
    --graph-degree 32 --build-l 64 --pq-chunks 128 \
    --bf-entries 99991

Recorded Deep1B QD64 build:
  script/bang_index_build.sh \
    --preset deep1b-qd64 \
    --base /path/to/deep/1B/base.1B.fbin \
    --dataset-name deep1b \
    --output-dir /large-local-storage/bang/deep1b \
    --build-disk-index /path/to/DiskANN/apps/build_disk_index

Important:
  Index construction is CPU/RAM/storage intensive. The recorded Deep1B run
  produced about 683 GB disk.index, 644 GB disk.bin, and 64 GB compressed PQ;
  allow substantial additional space for builder intermediates.
  --staging-dir must be on the same filesystem as --output-dir. A failed
  preserved stage can be passed back with the same --staging-dir argument.
  BANG search is a separate GPU step. Compile bang_search with MAX_R equal to
  --graph-degree and BF_ENTRIES equal to --bf-entries. BF_ENTRIES is recorded
  in metadata but does not change the index bytes.
EOF
}

task_fail() {
    echo "ERROR: $*" >&2
    exit 2
}

task_require_value() {
    local task_option="$1"
    local task_value="${2-}"
    [[ -n "$task_value" ]] || task_fail "$task_option requires a value"
}

task_require_positive_integer() {
    local task_name="$1"
    local task_value="$2"
    [[ "$task_value" =~ ^[1-9][0-9]*$ ]] ||
        task_fail "$task_name must be a positive integer: $task_value"
}

task_require_nonnegative_integer() {
    local task_name="$1"
    local task_value="$2"
    [[ "$task_value" =~ ^[0-9]+$ ]] ||
        task_fail "$task_name must be a non-negative integer: $task_value"
}

task_resolve_executable() {
    local task_value="$1"
    if [[ "$task_value" == */* ]]; then
        [[ -x "$task_value" ]] ||
            task_fail "executable is missing or not executable: $task_value"
        realpath -- "$task_value"
    else
        command -v "$task_value" ||
            task_fail "command not found: $task_value"
    fi
}

task_print_command() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
}

task_base=""
task_dataset=""
task_output_dir=""
task_preset=""
task_builder_api="pipeann"
task_graph_r=16
task_build_l=64
task_pq_chunks=128
task_build_memory_gb="${BANG_BUILD_MEMORY_GB:-64}"
task_search_dram_budget_gb=4
task_indexing_ram_budget_gb=32
task_build_pq_bytes=""
task_quantized_dim=""
task_pq_disk_bytes=0
task_threads="${BANG_BUILD_THREADS:-64}"
task_expect_points=0
task_bf_entries=399887
task_build_disk_index="${BANG_BUILD_DISK_INDEX:-build_disk_index}"
task_python="${BANG_PYTHON:-python3}"
task_require_normalized=1
task_staging_dir=""
task_keep_staging_on_failure=0
task_force=0
task_dry_run=0

task_raw_args=("$@")
for ((task_arg_i = 0; task_arg_i < ${#task_raw_args[@]}; task_arg_i++)); do
    if [[ "${task_raw_args[$task_arg_i]}" == "--preset" ]]; then
        ((task_arg_i + 1 < ${#task_raw_args[@]})) ||
            task_fail "--preset requires a value"
        [[ -z "$task_preset" ]] ||
            task_fail "--preset may be specified only once"
        task_preset="${task_raw_args[$((task_arg_i + 1))]}"
        ((task_arg_i += 1))
    fi
done

case "$task_preset" in
    "")
        ;;
    deep1b-qd64)
        task_builder_api="diskann"
        task_graph_r=64
        task_build_l=100
        task_pq_chunks=64
        task_search_dram_budget_gb=4
        task_indexing_ram_budget_gb=32
        task_build_pq_bytes=64
        task_quantized_dim=64
        task_pq_disk_bytes=0
        task_threads=24
        task_expect_points=1000000000
        task_require_normalized=0
        task_keep_staging_on_failure=1
        ;;
    *)
        task_fail "unknown preset: $task_preset"
        ;;
esac

while (($# > 0)); do
    case "$1" in
        --base)
            task_require_value "$1" "${2-}"
            task_base="$2"
            shift 2
            ;;
        --dataset-name)
            task_require_value "$1" "${2-}"
            task_dataset="$2"
            shift 2
            ;;
        --output-dir)
            task_require_value "$1" "${2-}"
            task_output_dir="$2"
            shift 2
            ;;
        --preset)
            task_require_value "$1" "${2-}"
            shift 2
            ;;
        --builder-api)
            task_require_value "$1" "${2-}"
            task_builder_api="$2"
            shift 2
            ;;
        --graph-degree)
            task_require_value "$1" "${2-}"
            task_graph_r="$2"
            shift 2
            ;;
        --build-l)
            task_require_value "$1" "${2-}"
            task_build_l="$2"
            shift 2
            ;;
        --pq-chunks)
            task_require_value "$1" "${2-}"
            task_pq_chunks="$2"
            shift 2
            ;;
        --build-memory-gb)
            task_require_value "$1" "${2-}"
            task_build_memory_gb="$2"
            shift 2
            ;;
        --search-dram-budget-gb)
            task_require_value "$1" "${2-}"
            task_search_dram_budget_gb="$2"
            shift 2
            ;;
        --indexing-ram-budget-gb)
            task_require_value "$1" "${2-}"
            task_indexing_ram_budget_gb="$2"
            shift 2
            ;;
        --build-pq-bytes)
            task_require_value "$1" "${2-}"
            task_build_pq_bytes="$2"
            shift 2
            ;;
        --quantized-dim)
            task_require_value "$1" "${2-}"
            task_quantized_dim="$2"
            shift 2
            ;;
        --pq-disk-bytes)
            task_require_value "$1" "${2-}"
            task_pq_disk_bytes="$2"
            shift 2
            ;;
        --threads)
            task_require_value "$1" "${2-}"
            task_threads="$2"
            shift 2
            ;;
        --expect-points)
            task_require_value "$1" "${2-}"
            task_expect_points="$2"
            shift 2
            ;;
        --bf-entries)
            task_require_value "$1" "${2-}"
            task_bf_entries="$2"
            shift 2
            ;;
        --build-disk-index)
            task_require_value "$1" "${2-}"
            task_build_disk_index="$2"
            shift 2
            ;;
        --python)
            task_require_value "$1" "${2-}"
            task_python="$2"
            shift 2
            ;;
        --allow-non-normalized)
            task_require_normalized=0
            shift
            ;;
        --require-normalized)
            task_require_normalized=1
            shift
            ;;
        --staging-dir)
            task_require_value "$1" "${2-}"
            task_staging_dir="$2"
            shift 2
            ;;
        --keep-staging-on-failure)
            task_keep_staging_on_failure=1
            shift
            ;;
        --discard-staging-on-failure)
            task_keep_staging_on_failure=0
            shift
            ;;
        --force)
            task_force=1
            shift
            ;;
        --dry-run)
            task_dry_run=1
            shift
            ;;
        -h|--help)
            task_usage
            exit 0
            ;;
        --*)
            task_fail "unknown option: $1"
            ;;
        *)
            task_fail "unexpected positional argument: $1"
            ;;
    esac
done

[[ -n "$task_base" ]] || task_fail "--base is required"
[[ -n "$task_dataset" ]] || task_fail "--dataset-name is required"
[[ -n "$task_output_dir" ]] || task_fail "--output-dir is required"
[[ "$task_dataset" =~ ^[A-Za-z0-9._-]+$ ]] ||
    task_fail "--dataset-name must match [A-Za-z0-9._-]+: $task_dataset"
[[ "$task_builder_api" == "pipeann" || "$task_builder_api" == "diskann" ]] ||
    task_fail "--builder-api must be pipeann or diskann: $task_builder_api"
task_require_positive_integer --graph-degree "$task_graph_r"
task_require_positive_integer --build-l "$task_build_l"
task_require_positive_integer --pq-chunks "$task_pq_chunks"
task_require_positive_integer --build-memory-gb "$task_build_memory_gb"
task_require_positive_integer \
    --search-dram-budget-gb "$task_search_dram_budget_gb"
task_require_positive_integer \
    --indexing-ram-budget-gb "$task_indexing_ram_budget_gb"
task_require_nonnegative_integer --pq-disk-bytes "$task_pq_disk_bytes"
task_require_positive_integer --threads "$task_threads"
task_require_nonnegative_integer --expect-points "$task_expect_points"
task_require_positive_integer --bf-entries "$task_bf_entries"
((task_build_l >= task_graph_r)) ||
    task_fail "--build-l must be >= --graph-degree"

if [[ -z "$task_build_pq_bytes" ]]; then
    task_build_pq_bytes="$task_pq_chunks"
fi
if [[ -z "$task_quantized_dim" ]]; then
    task_quantized_dim="$task_pq_chunks"
fi
task_require_positive_integer --build-pq-bytes "$task_build_pq_bytes"
task_require_positive_integer --quantized-dim "$task_quantized_dim"

task_python="$(task_resolve_executable "$task_python")"
task_build_disk_index="$(task_resolve_executable "$task_build_disk_index")"
command -v flock >/dev/null || task_fail "flock is required"
[[ -x /usr/bin/time ]] || task_fail "/usr/bin/time is required"

task_base="$(
    "$task_python" - "$task_base" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
task_output_dir="$(
    "$task_python" - "$task_output_dir" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

IFS=$'\t' read -r task_n_rows task_dim task_base_bytes task_norm_min task_norm_max < <(
    "$task_python" - "$task_base" "$task_require_normalized" <<'PY'
from pathlib import Path
import math
import os
import struct
import sys

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(f"numpy is required for FBIN validation: {error}")

path = Path(sys.argv[1])
require_normalized = bool(int(sys.argv[2]))
if not path.is_file():
    raise SystemExit(f"base file does not exist: {path}")
if path.stat().st_size < 8:
    raise SystemExit(f"base file is shorter than its FBIN header: {path}")
with path.open("rb") as handle:
    header = handle.read(8)
n_rows, dim = struct.unpack("<II", header)
if n_rows <= 0 or dim <= 0:
    raise SystemExit(f"invalid FBIN shape {n_rows} x {dim}: {path}")
expected_bytes = 8 + n_rows * dim * 4
actual_bytes = path.stat().st_size
if actual_bytes != expected_bytes:
    raise SystemExit(
        f"FBIN size mismatch: expected {expected_bytes}, got {actual_bytes}: {path}"
    )
vectors = np.memmap(
    path, dtype="<f4", mode="r", offset=8, shape=(n_rows, dim)
)
sample = np.asarray(vectors[: min(n_rows, 10000)])
if not np.isfinite(sample).all():
    raise SystemExit("the first 10000 base vectors contain NaN or infinity")
norms = np.linalg.norm(sample, axis=1)
norm_min = float(norms.min())
norm_max = float(norms.max())
if require_normalized and not (norm_min >= 0.999 and norm_max <= 1.001):
    raise SystemExit(
        "base vectors are not unit-normalized within [0.999, 1.001]: "
        f"sample range is [{norm_min:.9g}, {norm_max:.9g}]. "
        "Use the normalized reproduction input or explicitly pass "
        "--allow-non-normalized."
    )
print(n_rows, dim, actual_bytes, f"{norm_min:.9g}", f"{norm_max:.9g}", sep="\t")
PY
)

if ((task_expect_points > 0 && task_n_rows != task_expect_points)); then
    task_fail \
        "FBIN point-count mismatch: expected $task_expect_points, got $task_n_rows"
fi
if [[ "$task_builder_api" == "diskann" ]]; then
    ((task_build_pq_bytes <= task_dim)) ||
        task_fail \
            "--build-pq-bytes ($task_build_pq_bytes) cannot exceed dimension ($task_dim)"
    ((task_quantized_dim <= task_dim)) ||
        task_fail \
            "--quantized-dim ($task_quantized_dim) cannot exceed dimension ($task_dim)"
else
    ((task_pq_chunks <= task_dim)) ||
        task_fail \
            "--pq-chunks ($task_pq_chunks) cannot exceed dimension ($task_dim)"
fi

if [[ "$task_builder_api" == "diskann" ]]; then
    if [[ "$task_build_pq_bytes" == "$task_quantized_dim" ]]; then
        task_stem="${task_dataset}_R${task_graph_r}_L${task_build_l}_QD${task_quantized_dim}"
    else
        task_stem="${task_dataset}_R${task_graph_r}_L${task_build_l}_BPQ${task_build_pq_bytes}_QD${task_quantized_dim}"
    fi
else
    task_stem="${task_dataset}_R${task_graph_r}_Lb${task_build_l}_PQ${task_pq_chunks}"
fi
task_prefix="${task_output_dir}/${task_stem}"
task_bang_prefix="${task_prefix}_bang"
task_manifest="${task_output_dir}/${task_stem}.build.json"

task_set_build_command() {
    local task_command_prefix="$1"
    if [[ "$task_builder_api" == "diskann" ]]; then
        task_build_command=(
            "$task_build_disk_index"
            --data_type float
            --dist_fn l2
            --data_path "$task_base"
            --index_path_prefix "$task_command_prefix"
            -R "$task_graph_r"
            -L "$task_build_l"
            -B "$task_search_dram_budget_gb"
            -M "$task_indexing_ram_budget_gb"
            -T "$task_threads"
            --PQ_disk_bytes "$task_pq_disk_bytes"
            --build_PQ_bytes "$task_build_pq_bytes"
            --QD "$task_quantized_dim"
        )
    else
        task_build_command=(
            "$task_build_disk_index"
            float
            "$task_base"
            "$task_command_prefix"
            "$task_graph_r"
            "$task_build_l"
            "$task_pq_chunks"
            "$task_build_memory_gb"
            "$task_threads"
            l2
            pq
        )
    fi
}

task_set_build_command "$task_prefix"

printf 'BASE=%s\n' "$task_base"
printf 'BASE_SHAPE=%sx%s\n' "$task_n_rows" "$task_dim"
printf 'BASE_BYTES=%s\n' "$task_base_bytes"
printf 'SAMPLED_NORM_RANGE=%s..%s\n' "$task_norm_min" "$task_norm_max"
printf 'BUILDER_API=%s\n' "$task_builder_api"
printf 'PRESET=%s\n' "${task_preset:-none}"
printf 'INDEX_PREFIX=%s\n' "$task_prefix"
printf 'BANG_INDEX_PREFIX=%s\n' "$task_bang_prefix"
printf 'EXPECTED_BANG_DISK_BYTES=%s\n' \
    "$((task_n_rows * (task_dim * 4 + 4 + task_graph_r * 4)))"
if [[ "$task_builder_api" == "diskann" ]]; then
    printf 'EXPECTED_PQ_COMPRESSED_BYTES=%s\n' \
        "$((8 + task_n_rows * task_quantized_dim))"
fi
printf 'SEARCH_BINARY_CONTRACT=MAX_R=%s BF_ENTRIES=%s\n' \
    "$task_graph_r" "$task_bf_entries"
task_print_command "${task_build_command[@]}"

if ((task_dry_run == 1)); then
    printf 'STATUS=DRY_RUN\n'
    exit 0
fi

mkdir -p "$task_output_dir"
task_output_dir="$(realpath -- "$task_output_dir")"
task_prefix="${task_output_dir}/${task_stem}"
task_bang_prefix="${task_prefix}_bang"
task_manifest="${task_output_dir}/${task_stem}.build.json"

exec {task_lock_fd}>"${task_output_dir}/.${task_stem}.lock"
flock -n "$task_lock_fd" ||
    task_fail "another build holds the lock for $task_stem"

task_regular_files=(
    "${task_prefix}_disk.index"
    "${task_prefix}_disk.bin"
    "${task_prefix}_disk_metadata.bin"
    "${task_prefix}_pq_compressed.bin"
    "${task_prefix}_pq_pivots.bin"
    "${task_bang_prefix}_pq_pivots.bin"
    "$task_manifest"
)
if [[ "$task_builder_api" == "pipeann" ]]; then
    task_regular_files+=("${task_prefix}_disk.index.tags")
fi
task_link_files=(
    "${task_bang_prefix}_disk.bin"
    "${task_bang_prefix}_disk_metadata.bin"
    "${task_bang_prefix}_pq_compressed.bin"
)
task_link_targets=(
    "${task_prefix}_disk.bin"
    "${task_prefix}_disk_metadata.bin"
    "${task_prefix}_pq_compressed.bin"
)
task_log_files=(
    "${task_output_dir}/${task_stem}.build.log"
    "${task_output_dir}/${task_stem}.preprocess.log"
    "${task_output_dir}/${task_stem}.rewrap.log"
)

task_any_existing=0
task_cache_complete=1
for task_path in "${task_regular_files[@]}"; do
    if [[ -e "$task_path" || -L "$task_path" ]]; then
        task_any_existing=1
    fi
    [[ -f "$task_path" && -s "$task_path" ]] || task_cache_complete=0
done
for task_i in "${!task_link_files[@]}"; do
    task_path="${task_link_files[$task_i]}"
    task_target="${task_link_targets[$task_i]}"
    if [[ -e "$task_path" || -L "$task_path" ]]; then
        task_any_existing=1
    fi
    if [[ ! -L "$task_path" || ! -s "$task_path" ]]; then
        task_cache_complete=0
    elif [[ "$(realpath -- "$task_path")" != "$task_target" ]]; then
        task_cache_complete=0
    fi
done

if ((task_cache_complete == 1)); then
    if ! "$task_python" - "$task_manifest" "$task_base" \
        "$task_n_rows" "$task_dim" "$task_base_bytes" \
        "$task_graph_r" "$task_build_l" "$task_pq_chunks" \
        "$task_build_memory_gb" "$task_threads" "$task_builder_api" \
        "$task_search_dram_budget_gb" "$task_indexing_ram_budget_gb" \
        "$task_build_pq_bytes" "$task_quantized_dim" "$task_pq_disk_bytes" \
        "$task_expect_points" <<'PY'
import json
from pathlib import Path
import sys

(
    manifest_s,
    base_s,
    n_rows_s,
    dim_s,
    base_bytes_s,
    graph_r_s,
    build_l_s,
    pq_chunks_s,
    memory_s,
    threads_s,
    builder_api,
    search_dram_s,
    indexing_ram_s,
    build_pq_s,
    quantized_dim_s,
    pq_disk_s,
    expect_points_s,
) = sys.argv[1:]
payload = json.loads(Path(manifest_s).read_text(encoding="utf-8"))
expected = {
    "base": str(Path(base_s).resolve()),
    "n_rows": int(n_rows_s),
    "dimension": int(dim_s),
    "base_bytes": int(base_bytes_s),
    "graph_degree": int(graph_r_s),
    "build_l": int(build_l_s),
    "threads": int(threads_s),
}
if payload.get("builder_api", "pipeann") != builder_api:
    raise SystemExit(
        "cache metadata mismatch for builder_api: "
        f"expected {builder_api!r}, "
        f"got {payload.get('builder_api', 'pipeann')!r}"
    )
if builder_api == "diskann":
    expected.update(
        {
            "search_dram_budget_gb": int(search_dram_s),
            "indexing_ram_budget_gb": int(indexing_ram_s),
            "build_pq_bytes": int(build_pq_s),
            "quantized_dim": int(quantized_dim_s),
            "pq_disk_bytes": int(pq_disk_s),
            "expected_points": int(expect_points_s),
        }
    )
else:
    expected.update(
        {
            "pq_chunks": int(pq_chunks_s),
            "build_memory_gb": int(memory_s),
        }
    )
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"cache metadata mismatch for {key}: "
            f"expected {value!r}, got {payload.get(key)!r}"
        )
PY
    then
        task_cache_complete=0
    fi
fi

if ((task_cache_complete == 1 && task_force == 0)); then
    printf 'STATUS=CACHE_HIT\n'
    printf 'INDEX_PREFIX=%s\n' "$task_prefix"
    printf 'BANG_INDEX_PREFIX=%s\n' "$task_bang_prefix"
    printf 'SEARCH_BINARY_CONTRACT=MAX_R=%s BF_ENTRIES=%s\n' \
        "$task_graph_r" "$task_bf_entries"
    exit 0
fi

if ((task_any_existing == 1 && task_force == 0)); then
    task_fail \
        "existing files are incomplete or do not match this build; inspect them or pass --force"
fi

if [[ -n "$task_staging_dir" ]]; then
    mkdir -p "$task_staging_dir"
    task_stage_dir="$(realpath -- "$task_staging_dir")"
    [[ "$task_stage_dir" != "$task_output_dir" ]] ||
        task_fail "--staging-dir must not equal --output-dir"
else
    task_stage_dir="$(mktemp -d "${task_output_dir}/.${task_stem}.tmp.XXXXXX")"
fi
[[ "$(stat -c %d "$task_stage_dir")" == "$(stat -c %d "$task_output_dir")" ]] ||
    task_fail "--staging-dir and --output-dir must be on the same filesystem"

task_stage_marker="${task_stage_dir}/.bang-index-stage"
task_stage_identity="$task_stem|$task_builder_api|$task_base|$task_build_disk_index"
if [[ "$task_builder_api" == "diskann" ]]; then
    task_stage_identity+="|B=$task_search_dram_budget_gb|M=$task_indexing_ram_budget_gb"
    task_stage_identity+="|T=$task_threads|PQD=$task_pq_disk_bytes"
    task_stage_identity+="|BPQ=$task_build_pq_bytes|QD=$task_quantized_dim"
else
    task_stage_identity+="|MEM=$task_build_memory_gb|T=$task_threads"
    task_stage_identity+="|PQ=$task_pq_chunks"
fi
if [[ -e "$task_stage_marker" ]]; then
    [[ "$(<"$task_stage_marker")" == "$task_stage_identity" ]] ||
        task_fail "--staging-dir belongs to a different build: $task_stage_dir"
elif find "$task_stage_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    task_fail "--staging-dir is non-empty and has no valid stage marker: $task_stage_dir"
else
    printf '%s\n' "$task_stage_identity" > "$task_stage_marker"
fi

task_cleanup() {
    local task_cleanup_rc=$?
    trap - EXIT INT TERM
    if [[ -n "${task_stage_dir:-}" && -d "$task_stage_dir" ]]; then
        if ((task_cleanup_rc == 0 || task_keep_staging_on_failure == 0)); then
            rm -rf -- "$task_stage_dir"
        else
            printf 'STAGING_PRESERVED=%s\n' "$task_stage_dir" >&2
            printf 'RESUME_WITH=--staging-dir %q\n' "$task_stage_dir" >&2
        fi
    fi
    exit "$task_cleanup_rc"
}
trap task_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

task_stage_prefix="${task_stage_dir}/${task_stem}"
task_stage_bang_prefix="${task_stage_prefix}_bang"
task_stage_build_log="${task_stage_dir}/${task_stem}.build.log"
task_stage_preprocess_log="${task_stage_dir}/${task_stem}.preprocess.log"
task_stage_rewrap_log="${task_stage_dir}/${task_stem}.rewrap.log"
task_stage_manifest="${task_stage_dir}/${task_stem}.build.json"
task_started_epoch="$(date +%s)"
printf 'STAGING_DIR=%s\n' "$task_stage_dir"

task_set_build_command "$task_stage_prefix"
task_stage_build_command=("${task_build_command[@]}")
task_stage_builder_outputs=(
    "${task_stage_prefix}_disk.index"
    "${task_stage_prefix}_pq_compressed.bin"
    "${task_stage_prefix}_pq_pivots.bin"
)
if [[ "$task_builder_api" == "pipeann" ]]; then
    task_stage_builder_outputs+=("${task_stage_prefix}_disk.index.tags")
fi

task_stage_builder_complete=1
for task_path in "${task_stage_builder_outputs[@]}"; do
    [[ -f "$task_path" && -s "$task_path" ]] ||
        task_stage_builder_complete=0
done
if ((task_stage_builder_complete == 1)); then
    printf 'BUILDER_STAGE=CACHE_HIT\n' | tee -a "$task_stage_build_log"
else
    {
        task_print_command "${task_stage_build_command[@]}"
        /usr/bin/time -v "${task_stage_build_command[@]}"
    } 2>&1 | tee -a "$task_stage_build_log"
fi

{
    task_print_command \
        "$task_python" "<embedded-bang-preprocess>" \
        "${task_stage_prefix}_disk.index" "${task_stage_prefix}_disk.bin" \
        "$task_dim" 2 "$task_graph_r"
    "$task_python" - \
        "${task_stage_prefix}_disk.index" \
        "${task_stage_prefix}_disk.bin" \
        "$task_dim" 2 "$task_graph_r" <<'PY'
import os
import struct
import sys

SECTORLEN = 4096


def read_exact(handle, nbytes, what):
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(
            f"short read for {what}: expected {nbytes}, got {len(data)}"
        )
    return data


index_path, out_path, dim_s, datatype_s, degree_s = sys.argv[1:]
dim = int(dim_s)
datatype = int(datatype_s)
degree_bound = int(degree_s)
dtype_size = 4 if datatype == 2 else 1
metadata_path = out_path[:-4] + "_metadata" + out_path[-4:]

with open(index_path, "rb") as reader, open(out_path, "wb") as writer, open(
    metadata_path, "wb"
) as metadata_writer:
    read_exact(reader, 8, "metadata header")
    total_nodes = struct.unpack("<Q", read_exact(reader, 8, "total_nodes"))[0]
    num_dim = struct.unpack("<Q", read_exact(reader, 8, "num_dim"))[0]
    medoid_bytes = read_exact(reader, 8, "medoid")
    medoid = struct.unpack("<Q", medoid_bytes)[0]
    max_node_len_bytes = read_exact(reader, 8, "max_node_len")
    max_node_len = struct.unpack("<Q", max_node_len_bytes)[0]
    nodes_per_sector = struct.unpack(
        "<Q", read_exact(reader, 8, "nodes_per_sector")
    )[0]
    read_exact(reader, 24, "frozen-point metadata")
    metadata_filesize = struct.unpack(
        "<Q", read_exact(reader, 8, "filesize")
    )[0]
    actual_filesize = os.path.getsize(index_path)
    filesize = metadata_filesize or actual_filesize

    if dim != num_dim:
        raise ValueError(
            f"dimension mismatch: argument dim={dim}, index dim={num_dim}"
        )
    if filesize != actual_filesize:
        raise ValueError(
            f"index metadata filesize {filesize} differs from actual "
            f"filesize {actual_filesize}"
        )
    if filesize % SECTORLEN != 0:
        raise ValueError(
            f"file size {filesize} is not a multiple of {SECTORLEN}"
        )

    metadata_writer.write(medoid_bytes)
    metadata_writer.write(max_node_len_bytes)
    metadata_writer.write(struct.pack("<I", datatype))
    metadata_writer.write(struct.pack("<I", dim))
    metadata_writer.write(struct.pack("<I", degree_bound))

    nodes_read = 0
    sectors = filesize // SECTORLEN - 1
    for sector in range(sectors):
        reader.seek((sector + 1) * SECTORLEN)
        for _ in range(nodes_per_sector):
            if nodes_read == total_nodes:
                break
            writer.write(read_exact(reader, dtype_size * dim, "node vector"))
            degree_bytes = read_exact(reader, 4, "degree")
            degree = struct.unpack("<I", degree_bytes)[0]
            writer.write(degree_bytes)
            if degree == 0 or degree > degree_bound:
                raise ValueError(
                    f"invalid degree {degree} at node {nodes_read}"
                )
            neighbors_raw = read_exact(
                reader, 4 * degree, "neighbor array"
            )
            neighbors = struct.unpack("<" + "I" * degree, neighbors_raw)
            padding = read_exact(
                reader, 4 * (degree_bound - degree), "neighbor padding"
            )
            writer.write(struct.pack("<" + "I" * degree, *sorted(neighbors)))
            writer.write(padding)
            nodes_read += 1
            if nodes_read % 1000000 == 0:
                print(
                    f"BANG_PREPROCESS_PROGRESS={nodes_read}/{total_nodes}",
                    flush=True,
                )
        if nodes_read == total_nodes:
            break
    metadata_writer.write(struct.pack("<I", nodes_read))

if nodes_read != total_nodes:
    raise RuntimeError(
        f"expected {total_nodes} nodes, discovered {nodes_read}"
    )
print(
    f"nodes={nodes_read} dim={num_dim} medoid={medoid} "
    f"max_node_len={max_node_len} nodes_per_sector={nodes_per_sector} "
    f"metadata_filesize={metadata_filesize} actual_filesize={actual_filesize}"
)
PY
} 2>&1 | tee "$task_stage_preprocess_log"

{
    task_print_command \
        "$task_python" "<embedded-pq-rewrap>" \
        "${task_stage_prefix}_pq_pivots.bin" \
        "${task_stage_bang_prefix}_pq_pivots.bin"
    "$task_python" - \
        "${task_stage_prefix}_pq_pivots.bin" \
        "${task_stage_bang_prefix}_pq_pivots.bin" <<'PY'
import shutil
import struct
import sys

src, dst = sys.argv[1:]
shutil.copyfile(src, dst)
with open(src, "rb") as reader:
    header = reader.read(8)
    if len(header) != 8:
        raise EOFError("PQ pivots file is shorter than its header")
    n_offsets, n_cols = struct.unpack("<II", header)
    offsets_raw = reader.read(8 * n_offsets)
    if len(offsets_raw) != 8 * n_offsets:
        raise EOFError("PQ pivots offset table is truncated")
    offsets = list(struct.unpack("<" + "Q" * n_offsets, offsets_raw))
if n_cols != 1:
    raise ValueError(f"unexpected offset table columns: {n_cols}")
if n_offsets == 4:
    bang_offsets = offsets
elif n_offsets == 5:
    bang_offsets = [offsets[0], offsets[1], offsets[3], offsets[4]]
else:
    raise ValueError(f"unexpected number of PQ offsets: {n_offsets}")
with open(dst, "r+b") as writer:
    writer.seek(0)
    writer.write(struct.pack("<II", 4, 1))
    writer.write(struct.pack("<QQQQ", *bang_offsets))
print("source_offsets", offsets)
print("bang_offsets", bang_offsets)
PY
} 2>&1 | tee "$task_stage_rewrap_log"

if [[ "$task_builder_api" == "diskann" ]]; then
    task_expected_pq_width="$task_quantized_dim"
else
    task_expected_pq_width="$task_pq_chunks"
fi
"$task_python" - \
    "$task_stage_prefix" "$task_stage_bang_prefix" \
    "$task_n_rows" "$task_dim" "$task_graph_r" \
    "$task_expected_pq_width" <<'PY'
from pathlib import Path
import struct
import sys

prefix = Path(sys.argv[1])
bang_prefix = Path(sys.argv[2])
n_rows = int(sys.argv[3])
dim = int(sys.argv[4])
graph_r = int(sys.argv[5])
pq_width = int(sys.argv[6])

required = [
    Path(f"{prefix}_disk.index"),
    Path(f"{prefix}_disk.bin"),
    Path(f"{prefix}_disk_metadata.bin"),
    Path(f"{prefix}_pq_compressed.bin"),
    Path(f"{prefix}_pq_pivots.bin"),
    Path(f"{bang_prefix}_pq_pivots.bin"),
]
for path in required:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty build output: {path}")

disk_index = Path(f"{prefix}_disk.index")
if disk_index.stat().st_size % 4096 != 0:
    raise RuntimeError("DiskANN graph index is not sector-aligned")

disk_bin = Path(f"{prefix}_disk.bin")
expected_disk_bytes = n_rows * (dim * 4 + 4 + graph_r * 4)
if disk_bin.stat().st_size != expected_disk_bytes:
    raise RuntimeError(
        f"BANG disk.bin size mismatch: expected {expected_disk_bytes}, "
        f"got {disk_bin.stat().st_size}"
    )

pq_compressed = Path(f"{prefix}_pq_compressed.bin")
expected_pq_bytes = 8 + n_rows * pq_width
if pq_compressed.stat().st_size != expected_pq_bytes:
    raise RuntimeError(
        f"compressed PQ size mismatch: expected {expected_pq_bytes}, "
        f"got {pq_compressed.stat().st_size}"
    )
with pq_compressed.open("rb") as handle:
    pq_rows, pq_columns = struct.unpack("<II", handle.read(8))
if (pq_rows, pq_columns) != (n_rows, pq_width):
    raise RuntimeError(
        "compressed PQ header mismatch: "
        f"expected {(n_rows, pq_width)}, got {(pq_rows, pq_columns)}"
    )

metadata_path = Path(f"{prefix}_disk_metadata.bin")
metadata = metadata_path.read_bytes()
if len(metadata) != 32:
    raise RuntimeError(
        f"BANG metadata must be 32 bytes, got {len(metadata)}"
    )
_medoid, _max_node_len, datatype, metadata_dim, metadata_r, metadata_n = (
    struct.unpack("<QQIIII", metadata)
)
if (datatype, metadata_dim, metadata_r, metadata_n) != (
    2,
    dim,
    graph_r,
    n_rows,
):
    raise RuntimeError(
        "BANG metadata mismatch: "
        f"datatype={datatype}, dim={metadata_dim}, R={metadata_r}, "
        f"n={metadata_n}"
    )

with Path(f"{bang_prefix}_pq_pivots.bin").open("rb") as handle:
    n_offsets, n_cols = struct.unpack("<II", handle.read(8))
if (n_offsets, n_cols) != (4, 1):
    raise RuntimeError(
        f"BANG PQ pivots header mismatch: {(n_offsets, n_cols)}"
    )
print("STAGED_OUTPUT_VALIDATION=PASS")
PY

task_finished_epoch="$(date +%s)"
"$task_python" - \
    "$task_stage_manifest" "$task_base" "$task_n_rows" "$task_dim" \
    "$task_base_bytes" "$task_norm_min" "$task_norm_max" \
    "$task_dataset" "$task_graph_r" "$task_build_l" "$task_pq_chunks" \
    "$task_build_memory_gb" "$task_threads" "$task_bf_entries" \
    "$task_builder_api" "$task_preset" \
    "$task_search_dram_budget_gb" "$task_indexing_ram_budget_gb" \
    "$task_build_pq_bytes" "$task_quantized_dim" "$task_pq_disk_bytes" \
    "$task_expect_points" \
    "$task_build_disk_index" "$task_python" "$task_prefix" "$task_bang_prefix" \
    "$task_started_epoch" "$task_finished_epoch" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

(
    manifest_s,
    base_s,
    n_rows_s,
    dim_s,
    base_bytes_s,
    norm_min_s,
    norm_max_s,
    dataset,
    graph_r_s,
    build_l_s,
    pq_chunks_s,
    memory_s,
    threads_s,
    bf_entries_s,
    builder_api,
    preset,
    search_dram_s,
    indexing_ram_s,
    build_pq_s,
    quantized_dim_s,
    pq_disk_s,
    expect_points_s,
    builder_s,
    python_s,
    prefix_s,
    bang_prefix_s,
    started_s,
    finished_s,
) = sys.argv[1:]
payload = {
    "format": "bang-pq-index-v2",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "hostname": platform.node(),
    "base": str(Path(base_s).resolve()),
    "n_rows": int(n_rows_s),
    "dimension": int(dim_s),
    "base_bytes": int(base_bytes_s),
    "sampled_norm_min": float(norm_min_s),
    "sampled_norm_max": float(norm_max_s),
    "dataset_name": dataset,
    "graph_degree": int(graph_r_s),
    "build_l": int(build_l_s),
    "pq_chunks": int(pq_chunks_s),
    "build_memory_gb": int(memory_s),
    "threads": int(threads_s),
    "builder_api": builder_api,
    "preset": preset or None,
    "search_dram_budget_gb": int(search_dram_s),
    "indexing_ram_budget_gb": int(indexing_ram_s),
    "build_pq_bytes": int(build_pq_s),
    "quantized_dim": int(quantized_dim_s),
    "pq_disk_bytes": int(pq_disk_s),
    "expected_points": int(expect_points_s),
    "metric": "l2",
    "build_mode": "pq",
    "datatype": "float32",
    "build_disk_index": str(Path(builder_s).resolve()),
    "python": str(Path(python_s).resolve()),
    "index_prefix": prefix_s,
    "bang_index_prefix": bang_prefix_s,
    "bf_entries": int(bf_entries_s),
    "search_binary_contract": {
        "MAX_R": int(graph_r_s),
        "BF_ENTRIES": int(bf_entries_s),
    },
    "build_elapsed_seconds": int(finished_s) - int(started_s),
}
Path(manifest_s).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

if ((task_force == 1 && task_any_existing == 1)); then
    task_backup_dir="${task_output_dir}/.${task_stem}.backup-$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$task_backup_dir"
    for task_path in \
        "${task_regular_files[@]}" \
        "${task_link_files[@]}" \
        "${task_log_files[@]}"; do
        if [[ -e "$task_path" || -L "$task_path" ]]; then
            mv -- "$task_path" "$task_backup_dir/"
        fi
    done
    printf 'PREVIOUS_BUILD_BACKUP=%s\n' "$task_backup_dir"
fi

task_publish_names=(
    "${task_stem}_disk.index"
    "${task_stem}_disk.bin"
    "${task_stem}_disk_metadata.bin"
    "${task_stem}_pq_compressed.bin"
    "${task_stem}_pq_pivots.bin"
    "${task_stem}_bang_pq_pivots.bin"
    "${task_stem}.build.json"
    "${task_stem}.build.log"
    "${task_stem}.preprocess.log"
    "${task_stem}.rewrap.log"
)
if [[ "$task_builder_api" == "pipeann" ]]; then
    task_publish_names+=("${task_stem}_disk.index.tags")
fi
for task_name in "${task_publish_names[@]}"; do
    mv -- "${task_stage_dir}/${task_name}" "${task_output_dir}/${task_name}"
done

ln -s "${task_stem}_disk.bin" "${task_bang_prefix}_disk.bin"
ln -s \
    "${task_stem}_disk_metadata.bin" \
    "${task_bang_prefix}_disk_metadata.bin"
ln -s \
    "${task_stem}_pq_compressed.bin" \
    "${task_bang_prefix}_pq_compressed.bin"

for task_path in "${task_regular_files[@]}"; do
    [[ -f "$task_path" && -s "$task_path" ]] ||
        task_fail "published output is missing or empty: $task_path"
done
for task_i in "${!task_link_files[@]}"; do
    task_path="${task_link_files[$task_i]}"
    task_target="${task_link_targets[$task_i]}"
    [[ -L "$task_path" && -s "$task_path" ]] ||
        task_fail "published BANG link is invalid: $task_path"
    [[ "$(realpath -- "$task_path")" == "$task_target" ]] ||
        task_fail "published BANG link has the wrong target: $task_path"
done

[[ -f "$task_stage_marker" ]] ||
    task_fail "refusing to clean an unmarked staging directory: $task_stage_dir"
rm -rf -- "$task_stage_dir"
task_stage_dir=""
trap - EXIT INT TERM
printf 'STATUS=BUILT\n'
printf 'INDEX_PREFIX=%s\n' "$task_prefix"
printf 'BANG_INDEX_PREFIX=%s\n' "$task_bang_prefix"
printf 'MANIFEST=%s\n' "$task_manifest"
printf 'SEARCH_BINARY_CONTRACT=MAX_R=%s BF_ENTRIES=%s\n' \
    "$task_graph_r" "$task_bf_entries"
