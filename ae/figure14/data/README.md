# Figure 14 data test

## Environment

- Python 3 with NumPy and Matplotlib.
- Run from the repository root.

## Test

```bash
python3 ae/figure14/plot_figure14.py --check-only
```

## Expected output

The command exits with status 0 and prints `DATA_OK rows=168 latest_k5_k10_sim=84 direct_fpma=11 fpma_fallback=3 ...`. It validates `figure14_results.csv`, `k5_k10_latest_metrics.csv`, and the referenced test result. A full run writes the plot and summary under `ae/figure14/output/`.

Validate the complete YAML manifest separately with:

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
```
