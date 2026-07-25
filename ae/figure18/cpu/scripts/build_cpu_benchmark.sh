#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cpu_dir="$(cd -- "${script_dir}/.." && pwd)"
source_file="${cpu_dir}/src/hnswlib_1b_qps.cpp"
include_dir="${cpu_dir}/third_party/hnswlib"
output="${cpu_dir}/bin/hnswlib_1b_qps"
compiler="${CXX:-g++}"
dry_run=0

usage() {
  cat <<'EOF'
Usage: build_cpu_benchmark.sh [--output PATH] [--compiler CXX] [--dry-run]

Build the Figure 18 CPU hnswlib benchmark from the bundled source and headers.
The optimization flags match the historical 2026-06-15 build.
EOF
}

while (($#)); do
  case "$1" in
    --output)
      (($# >= 2)) || { echo "ERROR: --output requires a path" >&2; exit 2; }
      output="$2"
      shift 2
      ;;
    --compiler)
      (($# >= 2)) || { echo "ERROR: --compiler requires a command" >&2; exit 2; }
      compiler="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v "${compiler}" >/dev/null 2>&1 ||
  { echo "ERROR: compiler not found: ${compiler}" >&2; exit 1; }
[[ -r "${source_file}" ]] || { echo "ERROR: missing source: ${source_file}" >&2; exit 1; }
[[ -r "${include_dir}/hnswlib/hnswlib.h" ]] ||
  { echo "ERROR: missing bundled hnswlib headers" >&2; exit 1; }

compile=(
  "${compiler}"
  -std=c++17
  -O3
  -DNDEBUG
  -fopenmp
  -march=native
  -ffast-math
  -funroll-loops
  -I "${include_dir}"
  "${source_file}"
  -o "${output}"
  -lpthread
)

if ((dry_run)); then
  printf 'DRY-RUN'
  printf ' %q' "${compile[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname -- "${output}")"
"${compile[@]}"
[[ -x "${output}" ]] || { echo "ERROR: output is not executable: ${output}" >&2; exit 1; }
echo "${output}"
