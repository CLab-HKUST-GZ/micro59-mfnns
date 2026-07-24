#!/usr/bin/env bash
# Create ignored, machine-local links for the two canonical index builders.
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

task_usage() {
    cat <<'EOF'
Usage:
  GPU_Baseline/configure.sh [OPTIONS]
  GPU_Baseline/configure.sh --check

Options:
  --bang-builder FILE   Compatible PipeANN/DiskANN build_disk_index executable
  --cagra-python FILE   Python executable containing numpy, cupy, and cuvs
  --bang-source DIR     Optional local PipeANN/DiskANN source directory
  --cagra-source DIR    Optional local cuVS/CAGRA source/workspace directory
  --check               Validate links, imports, and canonical build scripts
  -h, --help            Show this help

Created links are machine-specific and ignored by Git.
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

task_link() {
    local task_kind="$1"
    local task_source="$2"
    local task_target="$3"
    local task_resolved

    if [[ "$task_kind" == file ]]; then
        [[ -f "$task_source" && -x "$task_source" ]] ||
            task_fail "executable does not exist: $task_source"
    else
        [[ -d "$task_source" ]] ||
            task_fail "source directory does not exist: $task_source"
    fi
    task_resolved="$(realpath -- "$task_source")"
    if [[ -e "$task_target" && ! -L "$task_target" ]]; then
        task_fail "refusing to replace a non-symlink: $task_target"
    fi
    ln -sfn -- "$task_resolved" "$task_target"
    printf 'LINKED=%s -> %s\n' "$task_target" "$task_resolved"
}

task_bang_builder=""
task_cagra_python=""
task_bang_source=""
task_cagra_source=""
task_check=0

while (($# > 0)); do
    case "$1" in
        --bang-builder)
            task_require_value "$1" "${2-}"
            task_bang_builder="$2"
            shift 2
            ;;
        --cagra-python)
            task_require_value "$1" "${2-}"
            task_cagra_python="$2"
            shift 2
            ;;
        --bang-source)
            task_require_value "$1" "${2-}"
            task_bang_source="$2"
            shift 2
            ;;
        --cagra-source)
            task_require_value "$1" "${2-}"
            task_cagra_source="$2"
            shift 2
            ;;
        --check)
            task_check=1
            shift
            ;;
        -h|--help)
            task_usage
            exit 0
            ;;
        *)
            task_fail "unknown option: $1"
            ;;
    esac
done

[[ -z "$task_bang_builder" ]] ||
    task_link file "$task_bang_builder" "${task_root}/BANG/build_disk_index"
[[ -z "$task_cagra_python" ]] ||
    task_link file "$task_cagra_python" "${task_root}/CAGRA/python"
[[ -z "$task_bang_source" ]] ||
    task_link dir "$task_bang_source" "${task_root}/BANG/source"
[[ -z "$task_cagra_source" ]] ||
    task_link dir "$task_cagra_source" "${task_root}/CAGRA/source"

if ((task_check == 0)); then
    if [[ -z "$task_bang_builder" &&
          -z "$task_cagra_python" &&
          -z "$task_bang_source" &&
          -z "$task_cagra_source" ]]; then
        task_usage
        exit 2
    fi
    exit 0
fi

[[ -x "${task_root}/BANG/index_build.sh" ]] ||
    task_fail "canonical BANG builder link is invalid"
[[ -x "${task_root}/CAGRA/index_build.sh" ]] ||
    task_fail "canonical CAGRA builder link is invalid"
[[ -x "${task_root}/BANG/build_disk_index" ]] ||
    task_fail "configure --bang-builder before --check"
[[ -x "${task_root}/CAGRA/python" ]] ||
    task_fail "configure --cagra-python before --check"

"${task_root}/BANG/index_build.sh" --help >/dev/null
"${task_root}/CAGRA/index_build.sh" --help >/dev/null
"${task_root}/CAGRA/python" - <<'PY'
import cupy
import cuvs
import numpy

print(
    "CAGRA_PYTHON_IMPORTS=PASS "
    f"numpy={numpy.__version__} "
    f"cupy={cupy.__version__} "
    f"cuvs={cuvs.__version__}"
)
PY
printf 'BANG_BUILD_DISK_INDEX=%s\n' \
    "$(realpath -- "${task_root}/BANG/build_disk_index")"
printf 'GPU_BASELINE_CONFIGURATION=PASS\n'
