#!/bin/bash
#SBATCH -J hnsw_d100e1k
#SBATCH -p i96m3tu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=02:00:00
#SBATCH -o /hpc2hdd/home/rmeng603/workspace/CPU_HNSW/memory/20260615/004_hnswlib_cpu_fresh_qps_recall/logs/deep-r100-extra-%j.out
#SBATCH -e /hpc2hdd/home/rmeng603/workspace/CPU_HNSW/memory/20260615/004_hnswlib_cpu_fresh_qps_recall/logs/deep-r100-extra-%j.err

set -euo pipefail

ROOT="/hpc2hdd/home/rmeng603/workspace/CPU_HNSW"
MEMDIR="${ROOT}/memory/20260615/004_hnswlib_cpu_fresh_qps_recall"
BIN="${ROOT}/bin/hnswlib_1b_qps"

cd "${ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export SLURM_CPU_BIND=none

bash "${ROOT}/scripts/build_hnswlib_1b_qps.sh"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-none}"
echo "[INFO] Start: $(date --iso-8601=seconds)"
echo "[INFO] Deep1B R@100 extra EF=1000, same fresh config"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-96}" --cpu-bind=none --exact \
  numactl --interleave=all "${BIN}" \
  --dataset deep1b_r100_fresh_n100_extra \
  --index /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/hnsw_index_Deep1B_norm_M16_ef500_dynprec.bin \
  --query /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/query_vectors_fp32_norm_n100_seed42.bin \
  --gt /hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/memory/20260610/001_recall100_deep1b_t2i1b_scaled_curves_nq100/data/gt_labels_deep1b_topk100_n100_seed42_from_official.bin \
  --k 100 \
  --limit-queries 100 \
  --warmup-queries 10000 \
  --query-repeats 500 \
  --trials-per-ef 3 \
  --omp-schedule static \
  --omp-chunk 64 \
  --ef-list 1000 \
  --threads-list 96 \
  --output "${MEMDIR}/results/deep1b_r100_fresh_n100_extra_ef1000.tsv"

echo "[INFO] End: $(date --iso-8601=seconds)"
