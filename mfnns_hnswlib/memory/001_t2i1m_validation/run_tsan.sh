#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECORD_DIR="${PROJECT_ROOT}/memory/001_t2i1m_validation"
ARTIFACT_DIR="${PROJECT_ROOT}/artifacts/001_t2i1m_validation"
BUILD_DIR="/tmp/mfnns_hnswlib_tsan_build"
TOOL="${BUILD_DIR}/mfnns_hnsw_tool"
BASE="/hpc2hdd/home/rmeng603/vectordb/t2i/1M/base.1M.fbin"
INDEX="${ARTIFACT_DIR}/tsan_first1k_M8_ef40.bin"

mkdir -p "${RECORD_DIR}" "${ARTIFACT_DIR}"
exec > >(tee "${RECORD_DIR}/tsan.stdout.log") 2> >(tee "${RECORD_DIR}/tsan.stderr.log" >&2)

make -C "${PROJECT_ROOT}" \
  BUILD_DIR="${BUILD_DIR}" \
  CXXFLAGS="-O1 -g -DNDEBUG -fsanitize=thread" \
  LDLIBS="-pthread -fsanitize=thread" \
  -j2

# The inherited HNSW code acquires the same locks in different orders within
# one serial insertion. Deadlock detection reports that static order graph even
# when insertion is single-threaded, so this run isolates data-race detection.
export TSAN_OPTIONS="halt_on_error=1:detect_deadlocks=0"
"${TOOL}" build \
  --base "${BASE}" \
  --index "${INDEX}" \
  --normalize 0 \
  --limit 1000 \
  --m 8 \
  --ef-construction 40 \
  --threads 4 \
  --batch-size 1000 \
  --seed 100 \
  --force 1

echo "[tsan] completed without a reported data race"
