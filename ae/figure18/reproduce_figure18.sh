#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/plot_figure18.py"

echo "Figure 18 reproduced under ae/figure18/output/"
