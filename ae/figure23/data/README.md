# Figure 23 data test

## Environment

- Python 3 with the packages in `ae/figure23/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure23/plot_figure23.py --check-only
(cd ae/figure23/data && sha256sum -c SHA256SUMS)
```

## Expected output

The plotter prints `CHECK_OK rows=21 yamls=21 ...`, and every checksum reports `OK`. Generated figures and the summary are under `ae/figure23/output/`.
