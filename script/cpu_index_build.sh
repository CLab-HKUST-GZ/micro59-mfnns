#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${REPOSITORY_ROOT}/mfnns_hnswlib"
DATA_ROOT_REL="${CPU_DATA_ROOT:-../../vectordb}"
BUILD_DIR_REL="${CPU_BUILD_DIR:-mfnns_hnswlib/build}"
THREADS="${CPU_INDEX_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
BATCH_SIZE="${CPU_INDEX_BATCH_SIZE:-100000}"
SEED="${CPU_INDEX_SEED:-100}"
FORCE="${CPU_INDEX_FORCE:-0}"
INDEX_ROOT="${PROJECT_ROOT}/cpu_index"

ALL_TARGETS=(
  deep10m/raw
  deep10m/normalized
  gist1m/raw
  gist1m/normalized
  glove2m/raw
  glove2m/normalized
  pubmed/raw
  sift1m/raw
  sift1m/normalized
  t2i1m/raw
  t2i1m/normalized
  w2v1m/raw
  w2v1m/normalized
  wiki1m/raw
  wiki1m/normalized
  deep1b/normalized
  t2i1b/normalized
)

usage() {
  cat <<'EOF'
Usage: script/cpu_index_build.sh [--dry-run] [--list] [TARGET ...]

Build one or more CPU HNSW indexes. With no TARGET argument, all 17 cache
targets are built sequentially: 15 CPU-scale variants plus Deep1B and T2I1B.

Datasets:
  deep10m gist1m glove2m pubmed sift1m t2i1m w2v1m wiki1m deep1b t2i1b

A dataset name selects all supported variants. A dataset/variant target selects
one, for example deep10m/normalized. PubMed supports raw only; the two 1B
datasets support normalized only.

Relative-path environment overrides:
  CPU_DATA_ROOT         Dataset root (default: ../../vectordb)
  CPU_BUILD_DIR         Compiler output directory
                        (default: mfnns_hnswlib/build)

Build settings:
  CPU_INDEX_THREADS     Normalization/insertion threads (default: allocation
                        CPU count, or 8 outside Slurm)
  CPU_INDEX_BATCH_SIZE  Streaming batch rows (default: 100000)
  CPU_INDEX_SEED        HNSW seed (default: 100)
  CPU_INDEX_FORCE       Replace existing indexes: 0 or 1 (default: 0)

The 1B builds require a large-memory compute node, substantial disk space, and
a long allocation. The repaired lock ordering supports parallel construction.
EOF
}

fail() {
  echo "error: $*" >&2
  exit 2
}

require_relative_path() {
  local name="$1"
  local value="$2"
  [[ -n "${value}" && "${value}" != /* ]] ||
    fail "${name} must be a non-empty path relative to the repository root: ${value}"
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] ||
    fail "${name} must be a positive integer: ${value}"
}

dataset_config() {
  local target="$1"
  DATASET="${target%%/*}"
  VARIANT="${target#*/}"
  [[ "${DATASET}" != "${VARIANT}" ]] ||
    fail "target must include a variant: ${target}"

  case "${VARIANT}" in
    raw)
      NORMALIZE=0
      ;;
    normalized)
      NORMALIZE=1
      ;;
    *)
      fail "unsupported variant in target: ${target}"
      ;;
  esac

  M=32
  EF_CONSTRUCTION=100
  case "${DATASET}" in
    deep10m)
      SOURCE_REL="deep/1M/base.10M.fbin"
      FORMAT="fbin"
      ;;
    gist1m)
      SOURCE_REL="gist/gist/gist_base.fvecs"
      FORMAT="fvecs"
      ;;
    glove2m)
      SOURCE_REL="glove/glove2.2m/glove2.2m_base.fvecs"
      FORMAT="fvecs"
      ;;
    pubmed)
      SOURCE_REL="pubmed/doc_vectors_norm.bin"
      FORMAT="fbin"
      [[ "${VARIANT}" == raw ]] ||
        fail "unsupported target: ${target} (PubMed supports raw only)"
      ;;
    sift1m)
      SOURCE_REL="sift/1M/sift/sift_base.fvecs"
      FORMAT="fvecs"
      ;;
    t2i1m)
      SOURCE_REL="t2i/1M/base.1M.fbin"
      FORMAT="fbin"
      ;;
    w2v1m)
      SOURCE_REL="w2v/word2vec/word2vec_base.fvecs"
      FORMAT="fvecs"
      ;;
    wiki1m)
      SOURCE_REL="wiki/wiki1m/base.1M.fbin"
      FORMAT="fbin"
      ;;
    deep1b)
      SOURCE_REL="deep/1B/base.1B.fbin"
      FORMAT="fbin"
      M=16
      EF_CONSTRUCTION=500
      [[ "${VARIANT}" == normalized ]] ||
        fail "unsupported target: ${target} (Deep1B supports normalized only)"
      ;;
    t2i1b)
      SOURCE_REL="t2i/1B/base.1B.fbin"
      FORMAT="fbin"
      M=16
      EF_CONSTRUCTION=500
      [[ "${VARIANT}" == normalized ]] ||
        fail "unsupported target: ${target} (T2I1B supports normalized only)"
      ;;
    *)
      fail "unsupported dataset: ${DATASET}"
      ;;
  esac
}

append_dataset_targets() {
  local dataset="$1"
  case "${dataset}" in
    deep10m|gist1m|glove2m|sift1m|t2i1m|w2v1m|wiki1m)
      TARGETS+=("${dataset}/raw" "${dataset}/normalized")
      ;;
    pubmed)
      TARGETS+=("pubmed/raw")
      ;;
    deep1b|t2i1b)
      TARGETS+=("${dataset}/normalized")
      ;;
    *)
      fail "unsupported dataset: ${dataset}"
      ;;
  esac
}

DRY_RUN=0
LIST_ONLY=0
SELECTED=()
while (($# > 0)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --list)
      LIST_ONLY=1
      ;;
    --help|-h)
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

require_relative_path CPU_DATA_ROOT "${DATA_ROOT_REL}"
require_relative_path CPU_BUILD_DIR "${BUILD_DIR_REL}"
require_positive_integer CPU_INDEX_THREADS "${THREADS}"
require_positive_integer CPU_INDEX_BATCH_SIZE "${BATCH_SIZE}"
require_positive_integer CPU_INDEX_SEED "${SEED}"
[[ "${FORCE}" == 0 || "${FORCE}" == 1 ]] ||
  fail "CPU_INDEX_FORCE must be 0 or 1: ${FORCE}"

TARGETS=()
if ((${#SELECTED[@]} == 0)); then
  TARGETS=("${ALL_TARGETS[@]}")
else
  for selection in "${SELECTED[@]}"; do
    if [[ "${selection}" == */* ]]; then
      dataset_config "${selection}"
      TARGETS+=("${selection}")
    else
      append_dataset_targets "${selection}"
    fi
  done
fi

for target in "${TARGETS[@]}"; do
  dataset_config "${target}"
done

if ((LIST_ONLY)); then
  printf '%-20s %-5s %-9s %-3s %-15s %s\n' \
    TARGET INPUT NORMALIZE M EF_CONSTRUCTION SOURCE
  for target in "${TARGETS[@]}"; do
    dataset_config "${target}"
    printf '%-20s %-5s %-9s %-3s %-15s %s\n' \
      "${target}" "${FORMAT}" "${NORMALIZE}" "${M}" \
      "${EF_CONSTRUCTION}" "${SOURCE_REL}"
  done
  exit 0
fi

DATA_ROOT="${REPOSITORY_ROOT}/${DATA_ROOT_REL}"
BUILD_DIR="${REPOSITORY_ROOT}/${BUILD_DIR_REL}"
TOOL="${BUILD_DIR}/mfnns_hnsw_tool"

if ((!DRY_RUN)); then
  make -C "${PROJECT_ROOT}" BUILD_DIR="${BUILD_DIR}" tool
fi

for target in "${TARGETS[@]}"; do
  dataset_config "${target}"
  base_path="${DATA_ROOT}/${SOURCE_REL}"
  index_path="${INDEX_ROOT}/${DATASET}/${VARIANT}/hnsw_index_M${M}_ef${EF_CONSTRUCTION}.bin"

  command=(
    "${TOOL}" build
    --base "${base_path}"
    --base-format "${FORMAT}"
    --index "${index_path}"
    --normalize "${NORMALIZE}"
    --m "${M}"
    --ef-construction "${EF_CONSTRUCTION}"
    --threads "${THREADS}"
    --insertion-threads "${THREADS}"
    --batch-size "${BATCH_SIZE}"
    --seed "${SEED}"
    --force "${FORCE}"
  )

  if ((DRY_RUN)); then
    printf '[dry-run]'
    printf ' %q' "${command[@]}"
    printf '\n'
    continue
  fi

  [[ -r "${base_path}" ]] || fail "missing readable dataset: ${base_path}"
  if [[ -s "${index_path}" && "${FORCE}" == 0 ]]; then
    echo "[cpu-index] skip existing ${target}: ${index_path}"
    continue
  fi

  mkdir -p "$(dirname "${index_path}")"
  echo "[cpu-index] build ${target}"
  "${command[@]}"
  "${TOOL}" inspect --index "${index_path}"
done
