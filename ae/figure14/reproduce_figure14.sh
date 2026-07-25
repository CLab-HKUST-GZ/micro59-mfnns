#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHONDONTWRITEBYTECODE=1 python3 "${script_dir}/plot_figure14.py"

echo "Figure 14 reproduced under ae/figure14/output/"
