#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/build_figure16_data.py"
python3 "${script_dir}/plot_figure16.py"

echo "Figure 16 reproduced under ae/figure16/output/"
