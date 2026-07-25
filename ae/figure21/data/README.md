# Figure 21 data test

## Environment

- Python 3, GNU `sha256sum`, and the packages in `ae/figure21/requirements.txt`.
- Run from the repository root.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ae/figure21/validate_figure21.py
(cd ae/figure21/data && sha256sum -c SHA256SUMS)
```

## Expected output

The validator prints `CHECK_OK sweep=243 ansmet=3 configs=246 global_max_cycle=618147`; every checksum reports `OK`. The generated plot data is `ae/figure21/output/figure21_plot_data.tsv`.
