# Figure 19 data test

## Environment

- Python 3 with the packages in `ae/figure19/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure19/plot_figure19.py --check-only
(cd ae/figure19/data && sha256sum -c SHA256SUMS)
```

## Expected output

Both commands exit with status 0, and every checksum reports `OK`. The validated plot output is `ae/figure19/output/figure19.pdf` and `ae/figure19/output/figure19.png`.
