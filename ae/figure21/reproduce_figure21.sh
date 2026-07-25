#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHONDONTWRITEBYTECODE=1 python3 "${script_dir}/plot_figure21.py"
PYTHONDONTWRITEBYTECODE=1 python3 "${script_dir}/validate_figure21.py"
(
  cd "${script_dir}/configs"
  sha256sum -c SHA256SUMS
)
(
  cd "${script_dir}/scripts/historical"
  sha256sum -c SHA256SUMS
)
(
  cd "${script_dir}/data"
  sha256sum -c SHA256SUMS
)

echo "Figure 21 reproduced under ae/figure21/output/"
