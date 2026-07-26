#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${REPOSITORY_ROOT}/mfnns_hnswlib"
DATA_ROOT_REL="${CPU_DATA_ROOT:-../../vectordb}"
BUILD_DIR_REL="${CPU_BUILD_DIR:-mfnns_hnswlib/build}"
INDEX_ROOT_REL="${CPU_INDEX_ROOT:-mfnns_hnswlib/cpu_index}"
THREADS="${CPU_INDEX_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
BATCH_SIZE="${CPU_INDEX_BATCH_SIZE:-100000}"
INDEX_SEED="${CPU_INDEX_SEED:-100}"
QUERY_COUNT="${CPU_QUERY_COUNT:-1000}"
QUERY_SEED="${CPU_QUERY_SEED:-42}"
GT_K_LIST="${CPU_GT_TOPKS:-5,10,100}"
FORCE="${CPU_INDEX_FORCE:-0}"

ALL_TARGETS=(
  deep10m
  gist1m
  glove2m
  pubmed
  sift1m
  t2i1m
  w2v1m
  wiki1m
  deep1b
  t2i1b
)

usage() {
  cat <<'EOF'
Usage: script/cpu_index_build.sh [--dry-run] [--list] [TARGET ...]

Build one or more complete normalized CPU artifacts. Each target produces:

  1. an HNSW index built only from L2-normalized base vectors;
  2. a persisted L2-normalized query sample; and
  3. exact L2 ground truth computed from the same normalized base/query space.

With no TARGET argument, all ten datasets are built sequentially.

Datasets:
  deep10m gist1m glove2m pubmed sift1m t2i1m w2v1m wiki1m deep1b t2i1b

Each target is a dataset name. Normalization is mandatory, including a
defensive normalization pass for PubMed's already-normalized source files.
The historical raw/normalized directory layer has been removed.

Relative-path environment overrides:
  CPU_DATA_ROOT         Dataset root (default: ../../vectordb)
  CPU_BUILD_DIR         Compiler output directory
                        (default: mfnns_hnswlib/build)
  CPU_INDEX_ROOT        Index/query/GT output root
                        (default: mfnns_hnswlib/cpu_index)

Build settings:
  CPU_INDEX_THREADS     Normalization/insertion threads (default: allocation
                        CPU count, or 8 outside Slurm)
  CPU_INDEX_BATCH_SIZE  Streaming batch rows (default: 100000)
  CPU_INDEX_SEED        HNSW seed (default: 100)
  CPU_QUERY_COUNT       Persisted query rows (default: 1000)
  CPU_QUERY_SEED        Deterministic query-selection seed (default: 42)
  CPU_GT_TOPKS          Comma-separated exact-GT widths (default: 5,10,100)
  CPU_INDEX_FORCE       Replace the complete index/query/GT bundle: 0 or 1
                        (default: 0)

The default filenames exactly match the Figure 14 YAMLs:
  query_vectors_n1000_seed42.bin
  gt_labels_topk{5,10,100}_n1000_seed42.bin

If a source has fewer queries than CPU_QUERY_COUNT (PubMed has 100), its
normalized query/GT rows are repeated deterministically to the requested size.
Existing indexes without matching L2-normalization metadata are rejected;
set CPU_INDEX_FORCE=1 to rebuild them under the enforced policy.

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

require_gt_k_list() {
  local value="$1"
  [[ "${value}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] ||
    fail "CPU_GT_TOPKS must be comma-separated positive integers: ${value}"
}

metadata_value() {
  local path="$1"
  local field="$2"
  awk -F '\t' -v key="${field}" '$1 == key { print $2 }' "${path}"
}

validate_index_cache() {
  local index_path="$1"
  local metadata_path="$2"
  local expected_m="$3"
  local expected_ef="$4"
  local expected_base="$5"
  local expected_format="$6"
  local expected_seed="$7"

  [[ -s "${index_path}" ]] ||
    fail "missing or empty cached index: ${index_path}"
  [[ -s "${metadata_path}" ]] ||
    fail "missing normalization metadata: ${metadata_path}"
  [[ "$(metadata_value "${metadata_path}" format)" == mfnns_hnsw_index_v1 ]] ||
    fail "unsupported index metadata format: ${metadata_path}"
  [[ "$(metadata_value "${metadata_path}" normalization)" == l2 ]] ||
    fail "cached index is not proven L2-normalized: ${metadata_path}"
  [[ "$(metadata_value "${metadata_path}" M)" == "${expected_m}" ]] ||
    fail "cached index M mismatch: ${metadata_path}"
  [[ "$(metadata_value "${metadata_path}" ef_construction)" == "${expected_ef}" ]] ||
    fail "cached index ef_construction mismatch: ${metadata_path}"
  [[ "$(metadata_value "${metadata_path}" base_path)" == "${expected_base}" ]] ||
    fail "cached index base source mismatch: ${metadata_path}"
  [[ "$(metadata_value "${metadata_path}" base_format)" == "${expected_format}" ]] ||
    fail "cached index base format mismatch: ${metadata_path}"
  [[ "$(metadata_value "${metadata_path}" seed)" == "${expected_seed}" ]] ||
    fail "cached index seed mismatch: ${metadata_path}"

  local inspection
  inspection="$("${TOOL}" inspect --index "${index_path}")"
  grep -Fqx "M=${expected_m}" <<<"${inspection}" ||
    fail "serialized index M mismatch: ${index_path}"
  grep -Fqx "ef_construction=${expected_ef}" <<<"${inspection}" ||
    fail "serialized index ef_construction mismatch: ${index_path}"
}

validate_bundle_correspondence() {
  local index_metadata="$1"
  local bundle_metadata="$2"
  [[ -s "${bundle_metadata}" ]] ||
    fail "missing normalized query/GT metadata: ${bundle_metadata}"
  local index_fingerprint bundle_fingerprint
  index_fingerprint="$(metadata_value "${index_metadata}" base_fingerprint_fnv1a64)"
  bundle_fingerprint="$(metadata_value "${bundle_metadata}" base_fingerprint_fnv1a64)"
  [[ -n "${index_fingerprint}" && "${index_fingerprint}" == "${bundle_fingerprint}" ]] ||
    fail "index and ground truth were not derived from the same base vectors"
  [[ "$(metadata_value "${bundle_metadata}" normalization)" == l2 ]] ||
    fail "query/GT bundle is not proven L2-normalized: ${bundle_metadata}"
  [[ "$(metadata_value "${bundle_metadata}" distance)" == l2 ]] ||
    fail "ground-truth distance policy mismatch: ${bundle_metadata}"
}

dataset_config() {
  local target="$1"
  [[ "${target}" != */* ]] ||
    fail "variants are no longer accepted; use the dataset name: ${target}"
  DATASET="${target}"
  NORMALIZE=1

  M=32
  EF_CONSTRUCTION=100
  case "${DATASET}" in
    deep10m)
      SOURCE_REL="deep/1M/base.10M.fbin"
      FORMAT="fbin"
      QUERY_REL="deep/query/query.public.10K.fbin"
      QUERY_FORMAT="fbin"
      ;;
    gist1m)
      SOURCE_REL="gist/gist/gist_base.fvecs"
      FORMAT="fvecs"
      QUERY_REL="gist/gist/gist_query.fvecs"
      QUERY_FORMAT="fvecs"
      ;;
    glove2m)
      SOURCE_REL="glove/glove2.2m/glove2.2m_base.fvecs"
      FORMAT="fvecs"
      QUERY_REL="glove/glove2.2m/glove2.2m_query.fvecs"
      QUERY_FORMAT="fvecs"
      ;;
    pubmed)
      SOURCE_REL="pubmed/doc_vectors_norm.bin"
      FORMAT="fbin"
      QUERY_REL="pubmed/query_vectors_norm.bin"
      QUERY_FORMAT="fbin"
      ;;
    sift1m)
      SOURCE_REL="sift/1M/sift/sift_base.fvecs"
      FORMAT="fvecs"
      QUERY_REL="sift/1M/sift/sift_query.fvecs"
      QUERY_FORMAT="fvecs"
      ;;
    t2i1m)
      SOURCE_REL="t2i/1M/base.1M.fbin"
      FORMAT="fbin"
      QUERY_REL="t2i/query/query.public.100K.fbin"
      QUERY_FORMAT="fbin"
      ;;
    w2v1m)
      SOURCE_REL="w2v/word2vec/word2vec_base.fvecs"
      FORMAT="fvecs"
      QUERY_REL="w2v/word2vec/word2vec_query.fvecs"
      QUERY_FORMAT="fvecs"
      ;;
    wiki1m)
      SOURCE_REL="wiki/wiki1m/base.1M.fbin"
      FORMAT="fbin"
      QUERY_REL="wiki/wiki1m/queries.fbin"
      QUERY_FORMAT="fbin"
      ;;
    deep1b)
      SOURCE_REL="deep/1B/base.1B.fbin"
      FORMAT="fbin"
      QUERY_REL="deep/query/query.public.10K.fbin"
      QUERY_FORMAT="fbin"
      M=16
      EF_CONSTRUCTION=500
      ;;
    t2i1b)
      SOURCE_REL="t2i/1B/base.1B.fbin"
      FORMAT="fbin"
      QUERY_REL="t2i/query/query.public.100K.fbin"
      QUERY_FORMAT="fbin"
      M=16
      EF_CONSTRUCTION=500
      ;;
    *)
      fail "unsupported dataset: ${DATASET}"
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
require_relative_path CPU_INDEX_ROOT "${INDEX_ROOT_REL}"
require_positive_integer CPU_INDEX_THREADS "${THREADS}"
require_positive_integer CPU_INDEX_BATCH_SIZE "${BATCH_SIZE}"
require_positive_integer CPU_INDEX_SEED "${INDEX_SEED}"
require_positive_integer CPU_QUERY_COUNT "${QUERY_COUNT}"
require_positive_integer CPU_QUERY_SEED "${QUERY_SEED}"
require_gt_k_list "${GT_K_LIST}"
[[ "${FORCE}" == 0 || "${FORCE}" == 1 ]] ||
  fail "CPU_INDEX_FORCE must be 0 or 1: ${FORCE}"

TARGETS=()
if ((${#SELECTED[@]} == 0)); then
  TARGETS=("${ALL_TARGETS[@]}")
else
  for selection in "${SELECTED[@]}"; do
    dataset_config "${selection}"
    TARGETS+=("${selection}")
  done
fi

for target in "${TARGETS[@]}"; do
  dataset_config "${target}"
done

if ((LIST_ONLY)); then
  printf '%-12s %-5s %-5s %-9s %-3s %-15s %s\n' \
    DATASET BASE QUERY NORMALIZE M EF_CONSTRUCTION SOURCES
  for target in "${TARGETS[@]}"; do
    dataset_config "${target}"
    printf '%-12s %-5s %-5s %-9s %-3s %-15s %s | %s\n' \
      "${target}" "${FORMAT}" "${QUERY_FORMAT}" "${NORMALIZE}" "${M}" \
      "${EF_CONSTRUCTION}" "${SOURCE_REL}" "${QUERY_REL}"
  done
  exit 0
fi

DATA_ROOT="${REPOSITORY_ROOT}/${DATA_ROOT_REL}"
BUILD_DIR="${REPOSITORY_ROOT}/${BUILD_DIR_REL}"
INDEX_ROOT="${REPOSITORY_ROOT}/${INDEX_ROOT_REL}"
TOOL="${BUILD_DIR}/mfnns_hnsw_tool"

if ((!DRY_RUN)); then
  make -C "${PROJECT_ROOT}" BUILD_DIR="${BUILD_DIR}" tool
fi

for target in "${TARGETS[@]}"; do
  dataset_config "${target}"
  base_path="${DATA_ROOT}/${SOURCE_REL}"
  query_source_path="${DATA_ROOT}/${QUERY_REL}"
  output_dir="${INDEX_ROOT}/${DATASET}"
  index_path="${INDEX_ROOT}/${DATASET}/hnsw_index_M${M}_ef${EF_CONSTRUCTION}.bin"
  index_metadata="${index_path}.metadata.tsv"

  build_command=(
    "${TOOL}" build
    --base "${base_path}"
    --base-format "${FORMAT}"
    --index "${index_path}"
    --normalize 1
    --m "${M}"
    --ef-construction "${EF_CONSTRUCTION}"
    --threads "${THREADS}"
    --insertion-threads "${THREADS}"
    --batch-size "${BATCH_SIZE}"
    --seed "${INDEX_SEED}"
    --force "${FORCE}"
    --metadata "${index_metadata}"
  )
  prepare_command=(
    "${TOOL}" prepare
    --base "${base_path}"
    --base-format "${FORMAT}"
    --queries "${query_source_path}"
    --query-format "${QUERY_FORMAT}"
    --output-dir "${output_dir}"
    --query-count "${QUERY_COUNT}"
    --seed "${QUERY_SEED}"
    --gt-k-list "${GT_K_LIST}"
    --normalize 1
    --threads "${THREADS}"
    --batch-size "${BATCH_SIZE}"
    --force "${FORCE}"
  )

  if ((DRY_RUN)); then
    printf '[dry-run:index]'
    printf ' %q' "${build_command[@]}"
    printf '\n'
    printf '[dry-run:query-gt]'
    printf ' %q' "${prepare_command[@]}"
    printf '\n'
    continue
  fi

  [[ -r "${base_path}" ]] || fail "missing readable dataset: ${base_path}"
  [[ -r "${query_source_path}" ]] ||
    fail "missing readable query source: ${query_source_path}"

  if ((FORCE == 0)) && [[ -e "${index_path}" || -e "${index_metadata}" ]]; then
    if [[ -s "${index_path}" && -s "${index_metadata}" ]]; then
      validate_index_cache \
        "${index_path}" "${index_metadata}" "${M}" "${EF_CONSTRUCTION}" \
        "${base_path}" "${FORMAT}" "${INDEX_SEED}"
      echo "[cpu-index] verified existing normalized ${target}: ${index_path}"
    else
      fail "partial or unverified index cache for ${target}; set CPU_INDEX_FORCE=1 to rebuild"
    fi
  else
    mkdir -p "${output_dir}"
    echo "[cpu-index] build normalized ${target}"
    "${build_command[@]}"
    validate_index_cache \
      "${index_path}" "${index_metadata}" "${M}" "${EF_CONSTRUCTION}" \
      "${base_path}" "${FORMAT}" "${INDEX_SEED}"
  fi

  echo "[cpu-index] prepare normalized query and exact GT for ${target}"
  "${prepare_command[@]}"
  bundle_metadata="${output_dir}/bundle_metadata_n${QUERY_COUNT}_seed${QUERY_SEED}.tsv"
  validate_bundle_correspondence "${index_metadata}" "${bundle_metadata}"
  echo "[cpu-index] verified normalized index/query/GT correspondence for ${target}"
done
