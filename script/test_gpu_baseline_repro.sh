#!/usr/bin/env bash
# Fast regression for the frozen manifests and raw-data converter.
set -euo pipefail

task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${PYTHON:-python3}"
task_temp="$(mktemp -d /tmp/figure14_gpu_repro_test.XXXXXX)"
task_cleanup() {
    rm -rf -- "$task_temp"
}
trap task_cleanup EXIT INT TERM

"$task_python" - "$task_repo" "$task_temp" <<'PY'
import csv
from pathlib import Path
import struct
import sys

import numpy as np

repo = Path(sys.argv[1])
temp = Path(sys.argv[2])
params = repo / "GPU_Baseline" / "params"
cagra = list(csv.DictReader((params / "cagra.csv").open(newline="")))
bang = list(csv.DictReader((params / "bang.csv").open(newline="")))
contracts = list(csv.DictReader((params / "bang_contracts.csv").open(newline="")))
assert len(cagra) == 21
assert len(bang) == 21
assert len(contracts) == 4
assert {int(row["top_k"]) for row in cagra} == {5, 10, 100}
assert {int(row["top_k"]) for row in bang} == {5, 10, 100}
assert {(int(row["max_r"]), int(row["bf_entries"])) for row in contracts} == {
    (16, 399887), (32, 399887), (32, 99991), (64, 399887)
}
assert {row["cuda_version"] for row in contracts} == {"12.8"}

rng = np.random.default_rng(7)
raw = temp / "raw" / "sift" / "1M" / "sift"
raw.mkdir(parents=True)
base = rng.normal(size=(1200, 128)).astype(np.float32)
query = rng.normal(size=(1000, 128)).astype(np.float32)
for path, values in (
    (raw / "sift_base.fvecs", base),
    (raw / "sift_query.fvecs", query),
):
    with path.open("wb") as handle:
        for row in values:
            handle.write(struct.pack("<I", row.size))
            handle.write(row.astype("<f4", copy=False).tobytes())
PY

for task_profile in cagra bang; do
    "$task_python" "$task_repo/GPU_Baseline/prepare_data.py" \
        --profile "$task_profile" \
        --dataset sift1M \
        --raw-root "$task_temp/raw" \
        --output-root "$task_temp/output" \
        --gt-backend numpy --checksum
done

"$task_python" - "$task_temp/output" <<'PY'
from pathlib import Path
import json
import struct
import sys

root = Path(sys.argv[1])
expected = {
    root / "cagra" / "sift1m" / "base.fbin": (1200, 128),
    root / "cagra" / "sift1m" / "query.fbin": (1000, 128),
    root / "cagra" / "sift1m" / "gt_top100.ibin": (1000, 100),
    root / "bang" / "sift1M" / "base.fbin": (1200, 128),
    root / "bang" / "sift1M" / "query.fbin": (1000, 128),
    root / "bang" / "sift1M" / "gt_top100.bang.bin": (1000, 100),
}
for path, shape in expected.items():
    with path.open("rb") as handle:
        actual = struct.unpack("<II", handle.read(8))
    assert actual == shape, (path, actual, shape)
for path in (
    root / "cagra" / "sift1m" / "prepare.json",
    root / "bang" / "sift1M" / "prepare.json",
):
    payload = json.loads(path.read_text())
    assert payload["base_sha256"]
    assert payload["query_sha256"]
    assert payload["query_indices_sha256"]
    assert payload["gt_top100_sha256"]
print("TEST_GPU_BASELINE_REPRO=PASS")
PY
