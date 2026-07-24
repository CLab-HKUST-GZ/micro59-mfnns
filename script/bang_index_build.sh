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
    [--graph-degree N] \
    [--build-l N] \
    [--pq-chunks N] \
    [--build-memory-gb N] \
    [--threads N] \
    [--bf-entries N] \
    [--build-disk-index FILE] \
    [--python PYTHON] \
    [--allow-non-normalized] \
    [--force] \
    [--dry-run]

Input:
  Little-endian FBIN: uint32 n, uint32 dim, then exactly n * dim float32
  values. The reproduction datasets are unit-normalized; this is checked by
  default because their inner-product ground truth was reused for L2 search.

Build pipeline:
  1. PipeANN build_disk_index:
       build_disk_index float BASE PREFIX R L PQ MEMORY_GB THREADS l2 pq
  2. Convert PREFIX_disk.index to BANG's disk.bin + disk_metadata.bin.
  3. Rewrap PipeANN's PQ-pivots header and create the BANG prefix symlinks.

Defaults from the recorded seven-dataset R16 baseline:
  R=16, L=64, PQ=128, build memory=64 GiB, threads=64,
  BF_ENTRIES=399887. Deep10M used PQ=96.

The output prefixes match the recorded runs:
  DIR/NAME_R<R>_Lb<L>_PQ<PQ>
  DIR/NAME_R<R>_Lb<L>_PQ<PQ>_bang

Environment:
  BANG_BUILD_DISK_INDEX  PipeANN build_disk_index executable
                        (default: build_disk_index from PATH)
  BANG_PYTHON            Python with numpy (default: python3)
  BANG_BUILD_MEMORY_GB   Builder memory budget (default: 64)
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

Important:
  Index construction is CPU/RAM/storage intensive; use node-local SSD.
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
task_graph_r=16
task_build_l=64
task_pq_chunks=128
task_build_memory_gb="${BANG_BUILD_MEMORY_GB:-64}"
task_threads="${BANG_BUILD_THREADS:-64}"
task_bf_entries=399887
task_build_disk_index="${BANG_BUILD_DISK_INDEX:-build_disk_index}"
task_python="${BANG_PYTHON:-python3}"
task_require_normalized=1
task_force=0
task_dry_run=0

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
        --threads)
            task_require_value "$1" "${2-}"
            task_threads="$2"
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
task_require_positive_integer --graph-degree "$task_graph_r"
task_require_positive_integer --build-l "$task_build_l"
task_require_positive_integer --pq-chunks "$task_pq_chunks"
task_require_positive_integer --build-memory-gb "$task_build_memory_gb"
task_require_positive_integer --threads "$task_threads"
task_require_positive_integer --bf-entries "$task_bf_entries"
((task_build_l >= task_graph_r)) ||
    task_fail "--build-l must be >= --graph-degree"

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

((task_pq_chunks <= task_dim)) ||
    task_fail "--pq-chunks ($task_pq_chunks) cannot exceed dimension ($task_dim)"

task_stem="${task_dataset}_R${task_graph_r}_Lb${task_build_l}_PQ${task_pq_chunks}"
task_prefix="${task_output_dir}/${task_stem}"
task_bang_prefix="${task_prefix}_bang"
task_manifest="${task_output_dir}/${task_stem}.build.json"

task_build_command=(
    "$task_build_disk_index"
    float
    "$task_base"
    "$task_prefix"
    "$task_graph_r"
    "$task_build_l"
    "$task_pq_chunks"
    "$task_build_memory_gb"
    "$task_threads"
    l2
    pq
)

printf 'BASE=%s\n' "$task_base"
printf 'BASE_SHAPE=%sx%s\n' "$task_n_rows" "$task_dim"
printf 'BASE_BYTES=%s\n' "$task_base_bytes"
printf 'SAMPLED_NORM_RANGE=%s..%s\n' "$task_norm_min" "$task_norm_max"
printf 'INDEX_PREFIX=%s\n' "$task_prefix"
printf 'BANG_INDEX_PREFIX=%s\n' "$task_bang_prefix"
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
    "${task_prefix}_disk.index.tags"
    "${task_prefix}_disk.bin"
    "${task_prefix}_disk_metadata.bin"
    "${task_prefix}_pq_compressed.bin"
    "${task_prefix}_pq_pivots.bin"
    "${task_bang_prefix}_pq_pivots.bin"
    "$task_manifest"
)
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
        "$task_build_memory_gb" "$task_threads" <<'PY'
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
) = sys.argv[1:]
payload = json.loads(Path(manifest_s).read_text(encoding="utf-8"))
expected = {
    "base": str(Path(base_s).resolve()),
    "n_rows": int(n_rows_s),
    "dimension": int(dim_s),
    "base_bytes": int(base_bytes_s),
    "graph_degree": int(graph_r_s),
    "build_l": int(build_l_s),
    "pq_chunks": int(pq_chunks_s),
    "build_memory_gb": int(memory_s),
    "threads": int(threads_s),
}
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

task_stage_dir="$(mktemp -d "${task_output_dir}/.${task_stem}.tmp.XXXXXX")"
task_cleanup() {
    if [[ -n "${task_stage_dir:-}" && -d "$task_stage_dir" ]]; then
        rm -rf -- "$task_stage_dir"
    fi
}
trap task_cleanup EXIT INT TERM

task_stage_prefix="${task_stage_dir}/${task_stem}"
task_stage_bang_prefix="${task_stage_prefix}_bang"
task_stage_build_log="${task_stage_dir}/${task_stem}.build.log"
task_stage_preprocess_log="${task_stage_dir}/${task_stem}.preprocess.log"
task_stage_rewrap_log="${task_stage_dir}/${task_stem}.rewrap.log"
task_stage_manifest="${task_stage_dir}/${task_stem}.build.json"
task_started_epoch="$(date +%s)"

task_stage_build_command=(
    "$task_build_disk_index"
    float
    "$task_base"
    "$task_stage_prefix"
    "$task_graph_r"
    "$task_build_l"
    "$task_pq_chunks"
    "$task_build_memory_gb"
    "$task_threads"
    l2
    pq
)

{
    task_print_command "${task_stage_build_command[@]}"
    /usr/bin/time -v "${task_stage_build_command[@]}"
} 2>&1 | tee "$task_stage_build_log"

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
            neighbors = [
                struct.unpack("<I", read_exact(reader, 4, "neighbor"))[0]
                for _ in range(degree)
            ]
            padding = read_exact(
                reader, 4 * (degree_bound - degree), "neighbor padding"
            )
            writer.write(struct.pack("<" + "I" * degree, *sorted(neighbors)))
            writer.write(padding)
            nodes_read += 1
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

"$task_python" - \
    "$task_stage_prefix" "$task_stage_bang_prefix" \
    "$task_n_rows" "$task_dim" "$task_graph_r" <<'PY'
from pathlib import Path
import struct
import sys

prefix = Path(sys.argv[1])
bang_prefix = Path(sys.argv[2])
n_rows = int(sys.argv[3])
dim = int(sys.argv[4])
graph_r = int(sys.argv[5])

required = [
    Path(f"{prefix}_disk.index"),
    Path(f"{prefix}_disk.index.tags"),
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
    builder_s,
    python_s,
    prefix_s,
    bang_prefix_s,
    started_s,
    finished_s,
) = sys.argv[1:]
payload = {
    "format": "bang-pipeann-pq-index-v1",
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
    "${task_stem}_disk.index.tags"
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

rmdir -- "$task_stage_dir"
task_stage_dir=""
trap - EXIT INT TERM
printf 'STATUS=BUILT\n'
printf 'INDEX_PREFIX=%s\n' "$task_prefix"
printf 'BANG_INDEX_PREFIX=%s\n' "$task_bang_prefix"
printf 'MANIFEST=%s\n' "$task_manifest"
printf 'SEARCH_BINARY_CONTRACT=MAX_R=%s BF_ENTRIES=%s\n' \
    "$task_graph_r" "$task_bf_entries"
