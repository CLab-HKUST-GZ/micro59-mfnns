#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT_REL="${TABLE5_CACHE_ROOT:-../../MFANNS/recall_analysis/memory/vectordb_recall_20260320_n1000/cache}"
INDEX_ROOT_REL="${TABLE5_INDEX_ROOT:-cpu_index}"
OUTPUT_DIR_REL="${TABLE5_OUTPUT_DIR:-artifacts/table5_reproduction}"
BUILD_DIR_REL="${TABLE5_BUILD_DIR:-build}"
JOBS="${TABLE5_JOBS:-1}"
THREADS="${TABLE5_THREADS:-8}"
QUERY_LIMIT="${TABLE5_QUERY_LIMIT:-1000}"
K="${TABLE5_K:-10}"
EF="${TABLE5_EF:-500}"
RISK_RATIO="${TABLE5_RISK_RATIO:-1.0073}"

require_relative_path() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" || "${value}" == /* ]]; then
    echo "${name} must be a non-empty path relative to the repository root: ${value}" >&2
    exit 2
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer: ${value}" >&2
    exit 2
  fi
}

require_relative_path TABLE5_CACHE_ROOT "${CACHE_ROOT_REL}"
require_relative_path TABLE5_INDEX_ROOT "${INDEX_ROOT_REL}"
require_relative_path TABLE5_OUTPUT_DIR "${OUTPUT_DIR_REL}"
require_relative_path TABLE5_BUILD_DIR "${BUILD_DIR_REL}"

CACHE_ROOT="${PROJECT_ROOT}/${CACHE_ROOT_REL}"
INDEX_ROOT="${PROJECT_ROOT}/${INDEX_ROOT_REL}"
OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR_REL}"
BUILD_DIR="${PROJECT_ROOT}/${BUILD_DIR_REL}"

require_positive_integer TABLE5_JOBS "${JOBS}"
require_positive_integer TABLE5_THREADS "${THREADS}"
require_positive_integer TABLE5_QUERY_LIMIT "${QUERY_LIMIT}"
require_positive_integer TABLE5_K "${K}"
require_positive_integer TABLE5_EF "${EF}"

if (( K > EF )); then
  echo "TABLE5_K must not exceed TABLE5_EF" >&2
  exit 2
fi

DATASETS=(
  "t2i1m normalized"
  "wiki1m normalized"
  "glove2m normalized"
  "deep10m normalized"
  "pubmed raw"
  "w2v1m normalized"
  "sift1m normalized"
)

for spec in "${DATASETS[@]}"; do
  read -r dataset variant <<<"${spec}"
  cache_dir="${CACHE_ROOT}/${dataset}/${variant}"
  index_dir="${INDEX_ROOT}/${dataset}/${variant}"
  for required in \
    "${index_dir}/hnsw_index_M32_ef100.bin" \
    "${cache_dir}/query_vectors_n1000_seed42.bin" \
    "${cache_dir}/gt_labels_topk${K}_n1000_seed42.bin"; do
    if [[ ! -r "${required}" ]]; then
      echo "Missing readable Table 5 input: ${required}" >&2
      exit 2
    fi
  done
done

make -C "${PROJECT_ROOT}" BUILD_DIR="${BUILD_DIR}" table5
TOOL="${BUILD_DIR}/table5_dataset_runner"
DATASET_OUTPUT_DIR="${OUTPUT_DIR}/datasets"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${DATASET_OUTPUT_DIR}" "${LOG_DIR}"

pids=()
names=()

terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap terminate_children INT TERM ERR

wait_first() {
  local pid="${pids[0]}"
  local name="${names[0]}"
  if ! wait "${pid}"; then
    echo "Table 5 dataset failed: ${name}" >&2
    tail -n 80 "${LOG_DIR}/${name}.log" >&2 || true
    return 1
  fi
  echo "[table5] completed ${name}"
  pids=("${pids[@]:1}")
  names=("${names[@]:1}")
}

for spec in "${DATASETS[@]}"; do
  read -r dataset variant <<<"${spec}"
  name="${dataset}_${variant}"
  while (( ${#pids[@]} >= JOBS )); do
    wait_first
  done
  echo "[table5] starting ${dataset}/${variant}"
  (
    "${TOOL}" \
      --cache-root "${CACHE_ROOT}" \
      --index-root "${INDEX_ROOT}" \
      --dataset "${dataset}" \
      --variant "${variant}" \
      --output "${DATASET_OUTPUT_DIR}/${name}.csv" \
      --query-limit "${QUERY_LIMIT}" \
      --k "${K}" \
      --ef "${EF}" \
      --threads "${THREADS}" \
      --risk-ratio "${RISK_RATIO}"
  ) >"${LOG_DIR}/${name}.log" 2>&1 &
  pids+=("$!")
  names+=("${name}")
done

while (( ${#pids[@]} > 0 )); do
  wait_first
done
trap - INT TERM ERR

python3 "${PROJECT_ROOT}/table5_reproduction/aggregate_table5.py" \
  --input-dir "${DATASET_OUTPUT_DIR}" \
  --output-csv "${OUTPUT_DIR}/table5.csv" \
  --output-md "${OUTPUT_DIR}/table5.md" \
  --details-csv "${OUTPUT_DIR}/table5_details.csv"

echo "[table5] complete"
echo "[table5] table=${OUTPUT_DIR}/table5.csv"
echo "[table5] details=${OUTPUT_DIR}/table5_details.csv"
