#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECORD_DIR="${PROJECT_ROOT}/memory/001_t2i1m_validation"
ARTIFACT_DIR="${PROJECT_ROOT}/artifacts/001_t2i1m_validation"
BUILD_DIR="/tmp/mfnns_hnswlib_build"
TOOL="${BUILD_DIR}/mfnns_hnsw_tool"

BASE="/hpc2hdd/home/rmeng603/vectordb/t2i/1M/base.1M.fbin"
QUERIES="/hpc2hdd/home/rmeng603/vectordb/t2i/query/query.public.100K.fbin"
FULL_INDEX="/hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_t2i_mantissa/hnsw_index_text2img1M_norm_M32_ef100.bin"
SMOKE_INDEX="${ARTIFACT_DIR}/t2i1m_first20k_norm_M16_ef100.bin"
DETERMINISTIC_INDEX_A="${ARTIFACT_DIR}/t2i1m_first5k_single_thread_a.bin"
DETERMINISTIC_INDEX_B="${ARTIFACT_DIR}/t2i1m_first5k_single_thread_b.bin"

mkdir -p "${RECORD_DIR}" "${ARTIFACT_DIR}"
exec > >(tee "${RECORD_DIR}/validation.stdout.log") 2> >(tee "${RECORD_DIR}/validation.stderr.log" >&2)

echo "[environment] host=$(hostname)"
echo "[environment] date=$(date --iso-8601=seconds)"
echo "[environment] cpus=${SLURM_CPUS_PER_TASK:-16}"
g++ --version | head -n 1

make -C "${PROJECT_ROOT}" BUILD_DIR="${BUILD_DIR}" -j2

"${TOOL}" build \
  --base "${BASE}" \
  --index "${SMOKE_INDEX}" \
  --normalize 1 \
  --limit 20000 \
  --m 16 \
  --ef-construction 100 \
  --threads 16 \
  --batch-size 5000 \
  --seed 100 \
  --force 1

"${TOOL}" inspect --index "${SMOKE_INDEX}" \
  | tee "${RECORD_DIR}/smoke_index_header.txt"

"${TOOL}" evaluate \
  --base "${BASE}" \
  --queries "${QUERIES}" \
  --index "${SMOKE_INDEX}" \
  --normalize 1 \
  --query-limit 20 \
  --query-offset 0 \
  --k 10 \
  --ef 100 \
  --threads 16 \
  --batch-size 5000 \
  --precisions fp32,fp16_true,fp16_fpma,fp8_e4m3,int16,int8 \
  --output "${RECORD_DIR}/smoke_recall.csv"

"${TOOL}" inspect --index "${FULL_INDEX}" \
  | tee "${RECORD_DIR}/full_index_header.txt"

"${TOOL}" evaluate \
  --base "${BASE}" \
  --queries "${QUERIES}" \
  --index "${FULL_INDEX}" \
  --normalize 1 \
  --query-limit 10 \
  --query-offset 0 \
  --k 10 \
  --ef 100 \
  --threads 16 \
  --batch-size 100000 \
  --precisions fp32,fp16_true,fp16_fpma,fp8_e4m3,int16,int8 \
  --output "${RECORD_DIR}/full_t2i1m_recall.csv"

"${TOOL}" build \
  --base "${BASE}" \
  --index "${DETERMINISTIC_INDEX_A}" \
  --normalize 1 \
  --limit 5000 \
  --m 16 \
  --ef-construction 100 \
  --threads 1 \
  --batch-size 5000 \
  --seed 100 \
  --force 1

"${TOOL}" build \
  --base "${BASE}" \
  --index "${DETERMINISTIC_INDEX_B}" \
  --normalize 1 \
  --limit 5000 \
  --m 16 \
  --ef-construction 100 \
  --threads 1 \
  --batch-size 5000 \
  --seed 100 \
  --force 1

cmp "${DETERMINISTIC_INDEX_A}" "${DETERMINISTIC_INDEX_B}"
(
  cd "${ARTIFACT_DIR}"
  sha256sum \
    "$(basename "${DETERMINISTIC_INDEX_A}")" \
    "$(basename "${DETERMINISTIC_INDEX_B}")"
) | tee "${RECORD_DIR}/single_thread_determinism.sha256"

(
  cd "${ARTIFACT_DIR}"
  sha256sum "$(basename "${SMOKE_INDEX}")"
) > "${RECORD_DIR}/smoke_index.sha256"
echo "[validation] completed"
