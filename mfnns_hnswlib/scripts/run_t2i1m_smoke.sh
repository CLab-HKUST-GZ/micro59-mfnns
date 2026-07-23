#!/usr/bin/env bash
set -euo pipefail

: "${T2I_BASE_FBIN:?Set T2I_BASE_FBIN to base.1M.fbin}"
: "${T2I_QUERY_FBIN:?Set T2I_QUERY_FBIN to the T2I query FBIN}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${MFNNS_HNSW_BUILD_DIR:-/tmp/mfnns_hnswlib_build}"
OUTPUT_DIR="${MFNNS_HNSW_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/t2i1m_smoke}"
TOOL="${BUILD_DIR}/mfnns_hnsw_tool"
INDEX="${OUTPUT_DIR}/t2i_first20k_norm_M16_ef100.bin"

mkdir -p "${OUTPUT_DIR}"
make -C "${PROJECT_ROOT}" BUILD_DIR="${BUILD_DIR}" -j2

"${TOOL}" build \
  --base "${T2I_BASE_FBIN}" \
  --index "${INDEX}" \
  --normalize 1 \
  --limit 20000 \
  --m 16 \
  --ef-construction 100 \
  --threads 16 \
  --batch-size 5000 \
  --seed 100 \
  --force 1

"${TOOL}" inspect --index "${INDEX}"

"${TOOL}" evaluate \
  --base "${T2I_BASE_FBIN}" \
  --queries "${T2I_QUERY_FBIN}" \
  --index "${INDEX}" \
  --normalize 1 \
  --query-limit 20 \
  --k 10 \
  --ef 100 \
  --threads 16 \
  --batch-size 5000 \
  --output "${OUTPUT_DIR}/recall.csv"
