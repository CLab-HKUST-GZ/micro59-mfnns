# Figure 17 test

## Environment

- Python 3 with the packages in `ae/figure17/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure17/build_figure17_data.py --check-only
python3 ae/figure17/plot_figure17.py --check-only
bash ae/figure17/reproduce_figure17.sh
```

## Expected output

All commands exit with status 0. The reproduction prints `Figure 17 reproduced under ae/figure17/output/` and writes:

```text
ae/figure17/data/figure17_energy_breakdown.csv
ae/figure17/output/figure17_summary.tsv
ae/figure17/output/figure17.pdf
ae/figure17/output/figure17.png
```

## Inputs and YAMLs

Figure 17 uses the same fixed execution-trace table and final Recall@10 YAMLs
as Figure 16. `ae/energy_model.py` validates all 42 physical traces before
deriving the seven plotted logical methods. `NMP-Base-ET` and MFNNS share the
same trace and differ only in the compute-energy model.
