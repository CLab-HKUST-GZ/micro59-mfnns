# Figure 14 result test

## Environment

- Python 3 with NumPy, Matplotlib, and PyYAML.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
bash ae/validate_energy_figures.sh
```

## Expected output

The configuration pass criterion is 126 valid YAMLs; the synchronized baseline currently reports the documented `k=5/wiki1m/mfnns` manifest/YAML mismatch. The independent energy check exits with status 0 and ends with `Figures 16 and 17 validation passed.` New simulator statistics are written below `simulator/run_case/figure14_recall_gt0895/results/k<k>/<dataset>/`; the tracked energy trace remains at `simulator/run_case/figure14_recall_gt0895/results/paper_energy/k10/execution_traces.csv`.
