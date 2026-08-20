#!/usr/bin/env bash
set -euo pipefail

CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "${CASE_ROOT}/../../.." && pwd)"
MANIFEST="${CASE_ROOT}/manifests/final_cases.tsv"
VALIDATOR="${CASE_ROOT}/tools/validate_final_configs.py"
INPUT_CHECKER="${CASE_ROOT}/tools/check_final_inputs.py"
RUN_ONE="${CASE_ROOT}/scripts/run_final_case.sh"
SIMULATOR_BIN="${FIGURE14_SIMULATOR_BIN:-simulator/build/ramulator2}"
DRY_RUN=0
SKIP_EXISTING=0
SELECTED=()

usage() {
  cat <<'EOF'
Usage: run_final_cases.sh [--dry-run] [--skip-existing] [CONFIG ...]

With no CONFIG arguments, run all 126 Figure 14 simulator YAMLs in manifest
order. CONFIG paths must be repository-relative members of the final manifest.

Options:
  --dry-run        Validate the 126-YAML matrix and print the selected cases.
                   Generated index/query/GT files and ramulator2 are not needed.
  --skip-existing  Do not rerun a case whose configured stats file is nonempty.
  -h, --help       Show this help.

The runner is synchronous. Submit individual CONFIG paths through the cluster's
scheduler when running multiple cases concurrently.
EOF
}

fail() {
  echo "error: $*" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      fail "unknown option: $1"
      ;;
    *)
      SELECTED+=("$1")
      ;;
  esac
  shift
done

cd "${REPOSITORY_ROOT}"
python3 "${VALIDATOR}"

mapfile -t ALL_CONFIGS < <(awk -F $'\t' 'NR > 1 { print $9 }' "${MANIFEST}")
declare -A ALLOWED=()
for config in "${ALL_CONFIGS[@]}"; do
  ALLOWED["${config}"]=1
done

if ((${#SELECTED[@]} == 0)); then
  SELECTED=("${ALL_CONFIGS[@]}")
else
  for config in "${SELECTED[@]}"; do
    [[ "${config}" != /* && -n "${ALLOWED[${config}]+present}" ]] ||
      fail "config is not in the final manifest: ${config}"
  done
fi

if ((DRY_RUN)); then
  for config in "${SELECTED[@]}"; do
    echo "[figure14-batch] would run ${config}"
  done
  echo "FIGURE14_BATCH_DRY_RUN_OK cases=${#SELECTED[@]}"
  exit 0
fi

python3 "${INPUT_CHECKER}" "${SELECTED[@]}"
[[ "${SIMULATOR_BIN}" != /* && -x "${SIMULATOR_BIN}" ]] ||
  fail "missing repository-relative simulator executable: ${SIMULATOR_BIN}"

completed=0
skipped=0
for config in "${SELECTED[@]}"; do
  stat_path="$(awk '$1 == "stat_path:" { print $2; exit }' "${config}")"
  if ((SKIP_EXISTING)) && [[ -s "${stat_path}" ]]; then
    echo "[figure14-batch] skip existing ${config}"
    ((skipped += 1))
    continue
  fi
  echo "[figure14-batch] run ${config}"
  FIGURE14_INPUTS_PRECHECKED=1 "${RUN_ONE}" "${config}"
  ((completed += 1))
done

echo "FIGURE14_BATCH_OK completed=${completed} skipped=${skipped}"
