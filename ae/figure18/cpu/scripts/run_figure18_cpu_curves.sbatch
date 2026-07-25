#!/usr/bin/env bash
#SBATCH -J fig18_cpu
#SBATCH -p i96m3tu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=06:00:00
#SBATCH -o figure18-cpu-%j.out
#SBATCH -e figure18-cpu-%j.err

# Reproduce the CPU producer used by the Figure 18 DP1B/T2I1B curves.
set -euo pipefail

: "${DEEP_INDEX:?Set DEEP_INDEX to the Deep1B hnswlib index.}"
: "${DEEP_QUERY:?Set DEEP_QUERY to the 100-query Deep1B float cache.}"
: "${DEEP_GT_R10:?Set DEEP_GT_R10 to the Deep1B Recall@10 GT cache.}"
: "${DEEP_GT_R100:?Set DEEP_GT_R100 to the Deep1B Recall@100 GT cache.}"
: "${T2I_INDEX:?Set T2I_INDEX to the normalized-L2 T2I1B hnswlib index.}"
: "${T2I_QUERY:?Set T2I_QUERY to the normalized 100-query T2I1B cache.}"
: "${T2I_GT:?Set T2I_GT to the normalized-L2 T2I1B top-100 GT cache.}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cpu_dir="$(cd -- "${script_dir}/.." && pwd)"
binary="${CPU_BENCHMARK:-${cpu_dir}/bin/hnswlib_1b_qps}"
result_dir="${RESULT_DIR:-${cpu_dir}/results}"
mkdir -p "${result_dir}"

for input in \
  "${DEEP_INDEX}" "${DEEP_QUERY}" "${DEEP_GT_R10}" "${DEEP_GT_R100}" \
  "${T2I_INDEX}" "${T2I_QUERY}" "${T2I_GT}"; do
  [[ -s "${input}" ]] || { echo "ERROR: missing input: ${input}" >&2; exit 1; }
done

bash "${script_dir}/build_cpu_benchmark.sh" --output "${binary}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export SLURM_CPU_BIND=none

threads_list="${THREADS_LIST:-${SLURM_CPUS_PER_TASK:-96}}"
query_repeats="${QUERY_REPEATS:-500}"
trials_per_ef="${TRIALS_PER_EF:-3}"
warmup="${WARMUP:-10000}"
omp_schedule_kind="${OMP_SCHEDULE_KIND:-static}"
omp_chunk="${OMP_CHUNK:-64}"
numa_interleave="${NUMA_INTERLEAVE:-1}"

runner=(srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-96}" --cpu-bind=none --exact)
exec_prefix=()
if [[ "${numa_interleave}" == "1" ]]; then
  command -v numactl >/dev/null 2>&1 ||
    { echo "ERROR: NUMA_INTERLEAVE=1 but numactl is unavailable" >&2; exit 1; }
  exec_prefix=(numactl --interleave=all)
fi

run_curve() {
  local dataset="$1"
  local index="$2"
  local query="$3"
  local gt="$4"
  local k="$5"
  local ef_list="$6"
  local output="$7"

  "${runner[@]}" "${exec_prefix[@]}" "${binary}" \
    --dataset "${dataset}" \
    --index "${index}" \
    --query "${query}" \
    --gt "${gt}" \
    --k "${k}" \
    --limit-queries 100 \
    --warmup-queries "${warmup}" \
    --query-repeats "${query_repeats}" \
    --trials-per-ef "${trials_per_ef}" \
    --omp-schedule "${omp_schedule_kind}" \
    --omp-chunk "${omp_chunk}" \
    --ef-list "${ef_list}" \
    --threads-list "${threads_list}" \
    --output "${output}"
}

echo "[INFO] host=$(hostname -f) start=$(date --iso-8601=seconds)"
echo "[INFO] threads=${threads_list} repeats=${query_repeats} trials=${trials_per_ef}"

run_curve \
  deep1b_r10_fresh_n100 \
  "${DEEP_INDEX}" "${DEEP_QUERY}" "${DEEP_GT_R10}" 10 \
  "${DEEP_R10_EF_LIST:-10,12,20,40,100,200,300,500}" \
  "${result_dir}/deep1b_r10_fresh_n100.tsv"

run_curve \
  deep1b_r100_fresh_n100 \
  "${DEEP_INDEX}" "${DEEP_QUERY}" "${DEEP_GT_R100}" 100 \
  "${DEEP_R100_EF_LIST:-30,40,60,100,150,300,700}" \
  "${result_dir}/deep1b_r100_fresh_n100.tsv"

run_curve \
  t2i1b_norml2_r10_fresh_n100 \
  "${T2I_INDEX}" "${T2I_QUERY}" "${T2I_GT}" 10 \
  "${T2I_R10_EF_LIST:-30,40,60,100,150,300,1000,3000}" \
  "${result_dir}/t2i1b_norml2_r10_fresh_n100.tsv"

run_curve \
  t2i1b_norml2_r100_fresh_n100 \
  "${T2I_INDEX}" "${T2I_QUERY}" "${T2I_GT}" 100 \
  "${T2I_R100_EF_LIST:-60,100,150,300,500,1500,3000,5000}" \
  "${result_dir}/t2i1b_norml2_r100_fresh_n100.tsv"

echo "[INFO] end=$(date --iso-8601=seconds) results=${result_dir}"
