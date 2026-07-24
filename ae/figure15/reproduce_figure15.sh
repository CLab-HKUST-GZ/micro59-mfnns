#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/build_figure15_data.py"
python3 "${script_dir}/plot_figure15.py"

echo "Figure 15 reproduced under ae/figure15/output/"
