#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/build_figure17_data.py"
python3 "${script_dir}/plot_figure17.py"

echo "Figure 17 reproduced under ae/figure17/output/"
