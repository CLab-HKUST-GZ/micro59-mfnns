#!/bin/bash
#SBATCH -J hnsw_fresh
#SBATCH -p i96m3tu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=06:00:00
#SBATCH -o /hpc2hdd/home/rmeng603/workspace/CPU_HNSW/memory/20260615/004_hnswlib_cpu_fresh_qps_recall/logs/fresh-%j.out
#SBATCH -e /hpc2hdd/home/rmeng603/workspace/CPU_HNSW/memory/20260615/004_hnswlib_cpu_fresh_qps_recall/logs/fresh-%j.err

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
echo "[INFO] Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "[INFO] CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "[INFO] Start: $(date --iso-8601=seconds)"
echo "[INFO] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
awk '/Cpus_allowed_list/ {print "[INFO] Cpus_allowed_list=" $2}' /proc/$$/status

THREADS_LIST="${THREADS_LIST:-${SLURM_CPUS_PER_TASK:-96}}"
QUERY_REPEATS="${QUERY_REPEATS:-500}"
TRIALS_PER_EF="${TRIALS_PER_EF:-3}"
WARMUP="${WARMUP:-10000}"
OMP_SCHEDULE_KIND="${OMP_SCHEDULE_KIND:-static}"
OMP_CHUNK="${OMP_CHUNK:-64}"
NUMA_INTERLEAVE="${NUMA_INTERLEAVE:-1}"

RUNNER=(srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-96}" --cpu-bind=none --exact)
EXEC_PREFIX=()
if [[ "${NUMA_INTERLEAVE}" == "1" ]]; then
  EXEC_PREFIX=(numactl --interleave=all)
fi

echo "[INFO] THREADS_LIST=${THREADS_LIST}"
echo "[INFO] QUERY_REPEATS=${QUERY_REPEATS}"
echo "[INFO] TRIALS_PER_EF=${TRIALS_PER_EF}"
echo "[INFO] WARMUP=${WARMUP}"
echo "[INFO] OMP schedule: ${OMP_SCHEDULE_KIND}, chunk=${OMP_CHUNK}"
echo "[INFO] NUMA_INTERLEAVE=${NUMA_INTERLEAVE}"

if [[ "${RUN_DEEP_R10:-1}" == "1" ]]; then
  echo "[INFO] Run fresh Deep1B R@10"
  "${RUNNER[@]}" "${EXEC_PREFIX[@]}" "${BIN}" \
    --dataset deep1b_r10_fresh_n100 \
    --index /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/hnsw_index_Deep1B_norm_M16_ef500_dynprec.bin \
    --query /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/query_vectors_fp32_norm_n100_seed42.bin \
    --gt /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/gt_labels_1B_topk32_n100_seed42.bin \
    --k 10 \
    --limit-queries 100 \
    --warmup-queries "${WARMUP}" \
    --query-repeats "${QUERY_REPEATS}" \
    --trials-per-ef "${TRIALS_PER_EF}" \
    --omp-schedule "${OMP_SCHEDULE_KIND}" \
    --omp-chunk "${OMP_CHUNK}" \
    --ef-list "${DEEP_R10_EF_LIST:-10,12,20,40,100,200,300,500}" \
    --threads-list "${THREADS_LIST}" \
    --output "${MEMDIR}/results/deep1b_r10_fresh_n100.tsv"
fi

if [[ "${RUN_DEEP_R100:-1}" == "1" ]]; then
  echo "[INFO] Run fresh Deep1B R@100"
  "${RUNNER[@]}" "${EXEC_PREFIX[@]}" "${BIN}" \
    --dataset deep1b_r100_fresh_n100 \
    --index /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/hnsw_index_Deep1B_norm_M16_ef500_dynprec.bin \
    --query /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/query_vectors_fp32_norm_n100_seed42.bin \
    --gt /hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/memory/20260610/001_recall100_deep1b_t2i1b_scaled_curves_nq100/data/gt_labels_deep1b_topk100_n100_seed42_from_official.bin \
    --k 100 \
    --limit-queries 100 \
    --warmup-queries "${WARMUP}" \
    --query-repeats "${QUERY_REPEATS}" \
    --trials-per-ef "${TRIALS_PER_EF}" \
    --omp-schedule "${OMP_SCHEDULE_KIND}" \
    --omp-chunk "${OMP_CHUNK}" \
    --ef-list "${DEEP_R100_EF_LIST:-30,40,60,100,150,300,700}" \
    --threads-list "${THREADS_LIST}" \
    --output "${MEMDIR}/results/deep1b_r100_fresh_n100.tsv"
fi

if [[ "${RUN_T2I_R10:-1}" == "1" ]]; then
  echo "[INFO] Run fresh T2I1B normalized-L2 R@10"
  "${RUNNER[@]}" "${EXEC_PREFIX[@]}" "${BIN}" \
    --dataset t2i1b_norml2_r10_fresh_n100 \
    --index /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_t2i/hnsw_index_text2img1B_norm_M16_ef500_dynprec.bin \
    --query "${ROOT}/memory/20260609/002_t2i1b_normalized_l2_gt_qps/results/query_vectors_norm_n100_first.bin" \
    --gt "${ROOT}/memory/20260609/002_t2i1b_normalized_l2_gt_qps/results/t2i1b_norm_l2_gt_top100_n100_first.bin" \
    --k 10 \
    --limit-queries 100 \
    --warmup-queries "${WARMUP}" \
    --query-repeats "${QUERY_REPEATS}" \
    --trials-per-ef "${TRIALS_PER_EF}" \
    --omp-schedule "${OMP_SCHEDULE_KIND}" \
    --omp-chunk "${OMP_CHUNK}" \
    --ef-list "${T2I_R10_EF_LIST:-30,40,60,100,150,300,1000,3000}" \
    --threads-list "${THREADS_LIST}" \
    --output "${MEMDIR}/results/t2i1b_norml2_r10_fresh_n100.tsv"
fi

if [[ "${RUN_T2I_R100:-1}" == "1" ]]; then
  echo "[INFO] Run fresh T2I1B normalized-L2 R@100"
  "${RUNNER[@]}" "${EXEC_PREFIX[@]}" "${BIN}" \
    --dataset t2i1b_norml2_r100_fresh_n100 \
    --index /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_t2i/hnsw_index_text2img1B_norm_M16_ef500_dynprec.bin \
    --query "${ROOT}/memory/20260609/002_t2i1b_normalized_l2_gt_qps/results/query_vectors_norm_n100_first.bin" \
    --gt "${ROOT}/memory/20260609/002_t2i1b_normalized_l2_gt_qps/results/t2i1b_norm_l2_gt_top100_n100_first.bin" \
    --k 100 \
    --limit-queries 100 \
    --warmup-queries "${WARMUP}" \
    --query-repeats "${QUERY_REPEATS}" \
    --trials-per-ef "${TRIALS_PER_EF}" \
    --omp-schedule "${OMP_SCHEDULE_KIND}" \
    --omp-chunk "${OMP_CHUNK}" \
    --ef-list "${T2I_R100_EF_LIST:-60,100,150,300,500,1500,3000,5000}" \
    --threads-list "${THREADS_LIST}" \
    --output "${MEMDIR}/results/t2i1b_norml2_r100_fresh_n100.tsv"
fi

echo "[INFO] End: $(date --iso-8601=seconds)"
