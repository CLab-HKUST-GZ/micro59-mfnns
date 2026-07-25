# Figure 16 test

## Environment

- Python 3 with the packages in `ae/figure16/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure16/build_figure16_data.py --check-only
python3 ae/figure16/plot_figure16.py --check-only
bash ae/figure16/reproduce_figure16.sh
```

## Expected output

All commands exit with status 0. The reproduction prints `Figure 16 reproduced under ae/figure16/output/` and writes:

```text
ae/figure16/data/figure16_energy_efficiency.csv
ae/figure16/output/figure16_summary.tsv
ae/figure16/output/figure16.pdf
ae/figure16/output/figure16.png
```
