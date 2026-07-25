#!/bin/bash
set -euo pipefail

ROOT="/hpc2hdd/home/rmeng603/workspace/CPU_HNSW"
HNSWLIB_ROOT="/hpc2hdd/home/rmeng603/workspace/hnswlib"

mkdir -p "${ROOT}/bin"

g++ -std=c++17 -O3 -DNDEBUG -fopenmp -march=native -ffast-math -funroll-loops \
  -I "${HNSWLIB_ROOT}" \
  "${ROOT}/src/hnswlib_1b_qps.cpp" \
  -o "${ROOT}/bin/hnswlib_1b_qps" \
  -lpthread

echo "${ROOT}/bin/hnswlib_1b_qps"
