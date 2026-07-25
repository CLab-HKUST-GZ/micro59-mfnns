# Figure 22 data test

## Environment

- Python 3 with the packages in `ae/figure22/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure22/plot_figure22.py --check-only
(cd ae/figure22/data && sha256sum -c SHA256SUMS)
```

## Expected output

The plotter prints `CHECK_OK rows=14 ...`, and every checksum reports `OK`. Generated figures and the numeric summary are under `ae/figure22/output/`.
