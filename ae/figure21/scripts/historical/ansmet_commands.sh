#!/usr/bin/env bash
set -euo pipefail

ROOT=/hpc2hdd/home/rmeng603/workspace/MFANNS
EXP=/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/memory/20260329/ansmet_recall09_k10_efsearch
BUILD_DIR=/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/build_compute_sysgcc
RAMULATOR_BIN=/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/build_compute_sysgcc/ramulator2
RUN_ROOT="$EXP/runs/t2i1m_ef40_extend"
YAML="$EXP/yamls/t2i1m_normalized_k10_ef40.yaml"

# 2026-03-30 manual follow-up requested after the original 71-case sweep.
python3 "$ROOT/simulator/memory/run_yaml_case.py" \
  --launcher sbatch \
  --sbatch-wait \
  --partition i64m512u \
  --time-limit 01:00:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus-per-task 4 \
  --mem 128G \
  --exclude cpu1-73 \
  --skip-build \
  --build-dir "$BUILD_DIR" \
  --ramulator-bin "$RAMULATOR_BIN" \
  --result-root "$RUN_ROOT" \
  --job-name-prefix ak10x \
  "$YAML"
