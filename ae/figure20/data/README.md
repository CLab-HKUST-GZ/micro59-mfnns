# Figure 20 data test

## Environment

- Python 3 with the packages in `ae/figure20/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure20/plot_figure20.py --check-only
(cd ae/figure20/data && sha256sum -c SHA256SUMS)
```

## Expected output

The plotter prints the Figure 20 validation message, and every checksum reports `OK`. Generated numeric files are `ae/figure20/output/figure20_normalized.csv` and `ae/figure20/output/figure20_summary.tsv`.
