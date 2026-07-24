#!/usr/bin/env bash
set -euo pipefail

task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_builder="$task_repo/script/bang_index_build.sh"
task_tmp="$(mktemp -d)"
task_cleanup() {
    rm -rf -- "$task_tmp"
}
trap task_cleanup EXIT INT TERM

task_base="$task_tmp/base.fbin"
task_fake_builder="$task_tmp/build_disk_index"
python3 - "$task_base" "$task_fake_builder" <<'PY'
from pathlib import Path
import os
import stat
import struct
import sys

base = Path(sys.argv[1])
builder = Path(sys.argv[2])
n_rows = 64
dim = 8
with base.open("wb") as handle:
    handle.write(struct.pack("<II", n_rows, dim))
    vector = struct.pack("<" + "f" * dim, 1.0, *([0.0] * (dim - 1)))
    for _ in range(n_rows):
        handle.write(vector)

builder.write_text(
    r'''#!/usr/bin/env python3
from pathlib import Path
import math
import os
import struct
import sys

args = sys.argv[1:]
if args[0] == "float":
    base = Path(args[1])
    prefix = Path(args[2])
    graph_r = int(args[3])
    pq_width = int(args[5])
    write_tags = True
else:
    def value(option):
        return args[args.index(option) + 1]

    base = Path(value("--data_path"))
    prefix = Path(value("--index_path_prefix"))
    graph_r = int(value("-R"))
    pq_width = int(value("--QD"))
    write_tags = False

prefix.parent.mkdir(parents=True, exist_ok=True)
with base.open("rb") as handle:
    n_rows, dim = struct.unpack("<II", handle.read(8))
    vectors = [
        handle.read(dim * 4)
        for _ in range(n_rows)
    ]

if os.environ.get("FAKE_BUILDER_FAIL") == "1":
    Path(f"{prefix}_disk.index").write_bytes(b"partial")
    raise SystemExit(9)

node_len = dim * 4 + 4 + graph_r * 4
nodes_per_sector = 4096 // node_len
n_sectors = math.ceil(n_rows / nodes_per_sector)
file_size = (n_sectors + 1) * 4096
metadata = bytearray(4096)
struct.pack_into("<II", metadata, 0, 9, 1)
struct.pack_into(
    "<QQQQQ", metadata, 8,
    n_rows, dim, 0, node_len, nodes_per_sector
)
struct.pack_into("<QQQ", metadata, 48, 0, 0, 0)
struct.pack_into("<Q", metadata, 72, file_size)

with Path(f"{prefix}_disk.index").open("wb") as handle:
    handle.write(metadata)
    row = 0
    degree = min(graph_r, n_rows)
    for _ in range(n_sectors):
        sector = bytearray(4096)
        for slot in range(nodes_per_sector):
            if row == n_rows:
                break
            offset = slot * node_len
            sector[offset: offset + dim * 4] = vectors[row]
            struct.pack_into("<I", sector, offset + dim * 4, degree)
            neighbors = [(row + i + 1) % n_rows for i in range(degree)]
            struct.pack_into(
                "<" + "I" * degree,
                sector,
                offset + dim * 4 + 4,
                *neighbors,
            )
            row += 1
        handle.write(sector)

if write_tags:
    Path(f"{prefix}_disk.index.tags").write_bytes(
        struct.pack("<II", n_rows, 1) + b"\0" * (n_rows * 4)
    )
Path(f"{prefix}_pq_compressed.bin").write_bytes(
    struct.pack("<II", n_rows, pq_width) + b"\0" * (n_rows * pq_width)
)
Path(f"{prefix}_pq_pivots.bin").write_bytes(
    struct.pack("<IIQQQQ", 4, 1, 40, 44, 48, 52) + b"\0" * 12
)
''',
    encoding="utf-8",
)
builder.chmod(
    builder.stat().st_mode
    | stat.S_IXUSR
    | stat.S_IXGRP
    | stat.S_IXOTH
)
PY

task_diskann_out="$task_tmp/diskann"
"$task_builder" \
    --builder-api diskann \
    --base "$task_base" \
    --dataset-name tiny \
    --output-dir "$task_diskann_out" \
    --build-disk-index "$task_fake_builder" \
    --graph-degree 4 \
    --build-l 8 \
    --build-pq-bytes 4 \
    --quantized-dim 4 \
    --expect-points 64 \
    --threads 2 > "$task_tmp/diskann-build.log"
grep -qx 'STATUS=BUILT' "$task_tmp/diskann-build.log"
task_diskann_prefix="$task_diskann_out/tiny_R4_L8_QD4"
[[ -s "${task_diskann_prefix}_disk.index" ]]
[[ ! -e "${task_diskann_prefix}_disk.index.tags" ]]
[[ "$(stat -c %s "${task_diskann_prefix}_disk.bin")" == 3328 ]]
[[ "$(stat -c %s "${task_diskann_prefix}_pq_compressed.bin")" == 264 ]]

"$task_builder" \
    --builder-api diskann \
    --base "$task_base" \
    --dataset-name tiny \
    --output-dir "$task_diskann_out" \
    --build-disk-index "$task_fake_builder" \
    --graph-degree 4 \
    --build-l 8 \
    --build-pq-bytes 4 \
    --quantized-dim 4 \
    --expect-points 64 \
    --threads 2 \
    --bf-entries 99 > "$task_tmp/diskann-cache.log"
grep -qx 'STATUS=CACHE_HIT' "$task_tmp/diskann-cache.log"

task_pipeann_out="$task_tmp/pipeann"
"$task_builder" \
    --base "$task_base" \
    --dataset-name tiny \
    --output-dir "$task_pipeann_out" \
    --build-disk-index "$task_fake_builder" \
    --graph-degree 4 \
    --build-l 8 \
    --pq-chunks 4 \
    --threads 2 > "$task_tmp/pipeann-build.log"
grep -qx 'STATUS=BUILT' "$task_tmp/pipeann-build.log"
[[ -s "$task_pipeann_out/tiny_R4_Lb8_PQ4_disk.index.tags" ]]
python3 - "$task_pipeann_out/tiny_R4_Lb8_PQ4.build.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["format"] = "bang-pipeann-pq-index-v1"
for key in (
    "builder_api",
    "preset",
    "search_dram_budget_gb",
    "indexing_ram_budget_gb",
    "build_pq_bytes",
    "quantized_dim",
    "pq_disk_bytes",
    "expected_points",
):
    payload.pop(key, None)
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
"$task_builder" \
    --base "$task_base" \
    --dataset-name tiny \
    --output-dir "$task_pipeann_out" \
    --build-disk-index "$task_fake_builder" \
    --graph-degree 4 \
    --build-l 8 \
    --pq-chunks 4 \
    --threads 2 > "$task_tmp/pipeann-v1-cache.log"
grep -qx 'STATUS=CACHE_HIT' "$task_tmp/pipeann-v1-cache.log"

task_resume_out="$task_tmp/resume"
task_resume_stage="$task_resume_out/.preserved-stage"
mkdir -p "$task_resume_out"
if FAKE_BUILDER_FAIL=1 "$task_builder" \
    --builder-api diskann \
    --base "$task_base" \
    --dataset-name resume \
    --output-dir "$task_resume_out" \
    --staging-dir "$task_resume_stage" \
    --keep-staging-on-failure \
    --build-disk-index "$task_fake_builder" \
    --graph-degree 4 \
    --build-l 8 \
    --build-pq-bytes 4 \
    --quantized-dim 4 \
    --threads 2 > "$task_tmp/resume-fail.log" 2>&1; then
    echo "expected the fake builder failure" >&2
    exit 1
fi
[[ -f "$task_resume_stage/.bang-index-stage" ]]
grep -q "^STAGING_PRESERVED=$task_resume_stage$" "$task_tmp/resume-fail.log"

"$task_builder" \
    --builder-api diskann \
    --base "$task_base" \
    --dataset-name resume \
    --output-dir "$task_resume_out" \
    --staging-dir "$task_resume_stage" \
    --keep-staging-on-failure \
    --build-disk-index "$task_fake_builder" \
    --graph-degree 4 \
    --build-l 8 \
    --build-pq-bytes 4 \
    --quantized-dim 4 \
    --threads 2 > "$task_tmp/resume-success.log"
grep -qx 'STATUS=BUILT' "$task_tmp/resume-success.log"
[[ ! -e "$task_resume_stage" ]]

python3 - "$task_diskann_prefix.build.json" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["format"] == "bang-pq-index-v2"
assert payload["builder_api"] == "diskann"
assert payload["graph_degree"] == 4
assert payload["build_l"] == 8
assert payload["build_pq_bytes"] == 4
assert payload["quantized_dim"] == 4
assert payload["expected_points"] == 64
PY

echo "TEST_BANG_INDEX_BUILD=PASS"
