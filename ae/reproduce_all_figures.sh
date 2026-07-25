#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

echo "[Figure 14] validate 126 YAMLs and reproduce"
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
bash ae/figure14/reproduce_figure14.sh

for figure in 15 16 17; do
  echo "[Figure ${figure}] reproduce"
  bash "ae/figure${figure}/reproduce_figure${figure}.sh"
done

echo "[Figures 16--17] validate energy inputs and checksums"
bash ae/validate_energy_figures.sh

echo "[Figure 18] validate archived YAML/CPU/BANG evidence (no GPU execution)"
bash ae/figure18/validate_figure18.sh
bash ae/figure18/reproduce_figure18.sh

for figure in 19 20 21 22 23; do
  echo "[Figure ${figure}] reproduce"
  bash "ae/figure${figure}/reproduce_figure${figure}.sh"
done

echo "AE_OK figures=14-23 gpu_jobs=0 outputs=ae/figureNN/output/"
