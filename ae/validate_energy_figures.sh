#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/figure16/build_figure16_data.py" --check-only
python3 "${script_dir}/figure16/plot_figure16.py" --check-only
python3 "${script_dir}/figure17/build_figure17_data.py" --check-only
python3 "${script_dir}/figure17/plot_figure17.py" --check-only

(
  cd "${script_dir}/figure16/data"
  sha256sum -c SHA256SUMS
)
(
  cd "${script_dir}/figure17/data"
  sha256sum -c SHA256SUMS
)

echo "Figures 16 and 17 validation passed."
