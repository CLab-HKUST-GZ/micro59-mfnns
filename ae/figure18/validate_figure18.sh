#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

PYTHONDONTWRITEBYTECODE=1 python3 "${script_dir}/plot_figure18.py" --check-only
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${script_dir}/cpu/scripts/validate_cpu_provenance.py" --check-only
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${script_dir}/scripts/validate_bang_reference.py"

(
  cd "${script_dir}/data"
  sha256sum -c SHA256SUMS
)
(
  cd "${script_dir}/cpu/data"
  sha256sum -c SHA256SUMS
)

while IFS= read -r shell_file; do
  bash -n "${shell_file}"
done < <(
  find "${script_dir}" -type f \
    \( -name '*.sh' -o -name '*.sbatch' \) -print | sort
)

bash "${script_dir}/cpu/scripts/build_cpu_benchmark.sh" --dry-run >/dev/null

[[ -f "${repo_root}/simulator/memory/run_yaml_case.py" ]] ||
  { echo "ERROR: missing simulator YAML runner" >&2; exit 1; }
[[ -f "${repo_root}/script/simulator_build.sh" ]] ||
  { echo "ERROR: missing simulator build entry point" >&2; exit 1; }

echo "Figure 18 validation passed: plot, 108 YAMLs, CPU provenance/source, and BANG scripts."
