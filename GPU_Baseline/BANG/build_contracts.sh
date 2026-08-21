#!/usr/bin/env bash
# Build the four isolated BANG search binary contracts used by Figure 14.
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_contracts="${task_root}/params/bang_contracts.csv"
task_source=""
task_output=""
task_cuda_root="/usr/local/cuda-12.8"
task_jobs=8
task_force=0

task_usage() {
    cat <<'EOF'
Usage:
  GPU_Baseline/BANG/build_contracts.sh \
    --source /path/to/BANG_Base \
    --output /path/to/contracts \
    [--cuda-root /usr/local/cuda-12.8] [--jobs 8] [--force]

The source directory is copied into an isolated temporary directory for each
MAX_R/BF_ENTRIES pair. The supplied source tree is never modified. CUDA 12.8
is required because the recorded BANG source uses the pre-C++17 CMake setup
that fails with the CUDA 13.1 CUB headers.
EOF
}

task_fail() {
    echo "ERROR: $*" >&2
    exit 2
}

task_value() {
    [[ -n "${2-}" ]] || task_fail "$1 requires a value"
}

while (($#)); do
    case "$1" in
        --source) task_value "$1" "${2-}"; task_source="$2"; shift 2 ;;
        --output) task_value "$1" "${2-}"; task_output="$2"; shift 2 ;;
        --cuda-root) task_value "$1" "${2-}"; task_cuda_root="$2"; shift 2 ;;
        --jobs) task_value "$1" "${2-}"; task_jobs="$2"; shift 2 ;;
        --force) task_force=1; shift ;;
        -h|--help) task_usage; exit 0 ;;
        *) task_fail "unknown option: $1" ;;
    esac
done

[[ -d "$task_source" && -f "$task_source/bang_search.cu" ]] ||
    task_fail "--source must be a BANG_Base directory"
[[ -n "$task_output" ]] || task_fail "--output is required"
[[ -x "$task_cuda_root/bin/nvcc" ]] ||
    task_fail "CUDA compiler is missing: $task_cuda_root/bin/nvcc"
[[ "$task_jobs" =~ ^[1-9][0-9]*$ ]] || task_fail "--jobs must be positive"
task_cuda_version="$($task_cuda_root/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -1)"
[[ "$task_cuda_version" == "12.8" ]] ||
    task_fail "Figure 14 BANG contracts require CUDA 12.8, got $task_cuda_version"

task_source="$(realpath -- "$task_source")"
task_source_commit="$(git -C "$task_source" rev-parse HEAD 2>/dev/null || printf unknown)"
task_source_bang_sha256="$(sha256sum "$task_source/bang_search.cu" | awk '{print $1}')"
mkdir -p "$task_output"
task_output="$(realpath -- "$task_output")"
export PATH="$task_cuda_root/bin:$PATH"
export CUDA_HOME="$task_cuda_root"
export CUDACXX="$task_cuda_root/bin/nvcc"

while IFS=, read -r task_id task_max_r task_bf task_expected_cuda; do
    [[ "$task_id" != contract_id ]] || continue
    [[ "$task_expected_cuda" == "$task_cuda_version" ]] ||
        task_fail "contract $task_id expects CUDA $task_expected_cuda"
    task_contract_dir="$task_output/$task_id"
    task_cache_valid=0
    if ((task_force == 0)) &&
       [[ -x "$task_contract_dir/bang_search" &&
          -s "$task_contract_dir/libbang.so" &&
          -s "$task_contract_dir/bang_search.cu" &&
          -s "$task_contract_dir/build.json" &&
          -s "$task_contract_dir/SHA256SUMS" ]]; then
        if (
            cd "$task_contract_dir"
            sha256sum --status -c SHA256SUMS
        ) && python3 - "$task_contract_dir/build.json" "$task_id" \
            "$task_max_r" "$task_bf" "$task_cuda_version" \
            "$task_source_commit" "$task_source_bang_sha256" <<'PY'
import json
from pathlib import Path
import sys

path, contract, max_r, bf, cuda, source_commit, source_bang_sha256 = sys.argv[1:]
payload = json.loads(Path(path).read_text(encoding="utf-8"))
expected = {
    "contract_id": contract,
    "MAX_R": int(max_r),
    "BF_ENTRIES": int(bf),
    "cuda_version": cuda,
    "source_git_commit": source_commit,
    "source_bang_search_sha256": source_bang_sha256,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"contract cache mismatch for {key}: "
            f"expected {value!r}, got {payload.get(key)!r}"
        )
PY
        then
            task_cache_valid=1
        fi
    fi
    if ((task_cache_valid == 1)); then
        printf 'CONTRACT=%s STATUS=CACHE_HIT\n' "$task_id"
        continue
    fi
    task_job="$(mktemp -d "/tmp/figure14_bang_${task_id}.XXXXXX")"
    task_cleanup_contract() {
        rm -rf -- "$task_job"
    }
    trap task_cleanup_contract EXIT INT TERM
    mkdir -p "$task_job/source"
    cp -a "$task_source/." "$task_job/source/"
    rm -rf -- "$task_job/source/build"
    python3 - "$task_job/source/bang_search.cu" "$task_max_r" "$task_bf" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
max_r = sys.argv[2]
bf_entries = sys.argv[3]
text = path.read_text(encoding="utf-8")
text, count_r = re.subn(
    r"^#define MAX_R\s+\d+.*$",
    f"#define MAX_R {max_r} // Figure 14 graph-degree contract.",
    text,
    count=1,
    flags=re.MULTILINE,
)
text, count_bf = re.subn(
    r"^#define BF_ENTRIES\s+\d+U.*$",
    f"#define BF_ENTRIES  {bf_entries}U // Figure 14 Bloom-filter contract.",
    text,
    count=1,
    flags=re.MULTILINE,
)
if (count_r, count_bf) != (1, 1):
    raise SystemExit(
        f"macro rewrite failed: MAX_R matches={count_r}, BF_ENTRIES matches={count_bf}"
    )
path.write_text(text, encoding="utf-8")
PY
    grep -nE '^#define (MAX_R|BF_ENTRIES)' "$task_job/source/bang_search.cu"
    cmake -S "$task_job/source" -B "$task_job/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_TOOLKIT_ROOT_DIR="$task_cuda_root"
    cmake --build "$task_job/build" -j"$task_jobs"
    task_publish="$task_output/.${task_id}.tmp-$$"
    mkdir -p "$task_publish"
    cp -f "$task_job/build/bang_search" "$task_publish/bang_search"
    cp -f "$task_job/build/libbang.so" "$task_publish/libbang.so"
    cp -f "$task_job/source/bang_search.cu" "$task_publish/bang_search.cu"
    python3 - "$task_publish/build.json" "$task_id" "$task_max_r" \
        "$task_bf" "$task_cuda_version" "$task_source" \
        "$task_source_commit" "$task_source_bang_sha256" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

(
    path,
    contract,
    max_r,
    bf,
    cuda,
    source,
    source_commit,
    source_bang_sha256,
) = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "format": "figure14-bang-search-contract-v1",
            "contract_id": contract,
            "MAX_R": int(max_r),
            "BF_ENTRIES": int(bf),
            "cuda_version": cuda,
            "source": str(Path(source).resolve()),
            "source_git_commit": source_commit,
            "source_bang_search_sha256": source_bang_sha256,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
    (
        cd "$task_publish"
        sha256sum bang_search libbang.so bang_search.cu > SHA256SUMS
    )
    if [[ -d "$task_contract_dir" ]]; then
        task_backup="$task_output/.${task_id}.backup-$(date +%Y%m%dT%H%M%S)"
        mv -- "$task_contract_dir" "$task_backup"
        printf 'PREVIOUS_CONTRACT_BACKUP=%s\n' "$task_backup"
    fi
    mv -- "$task_publish" "$task_contract_dir"
    rm -rf -- "$task_job"
    trap - EXIT INT TERM
    printf 'CONTRACT=%s MAX_R=%s BF_ENTRIES=%s STATUS=BUILT\n' \
        "$task_id" "$task_max_r" "$task_bf"
done < "$task_contracts"

printf 'BANG_CONTRACTS=4 CUDA=%s STATUS=PASS\n' "$task_cuda_version"
